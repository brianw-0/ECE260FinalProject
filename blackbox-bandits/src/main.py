import torch as ch
from torchvision import models, transforms
#from torchvision.datasets import ImageFolder
from Dataset260 import ImageFolder260
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
import argparse
import json
import pdb

import time

# CLASSIFIERS = {
#     "inception_v3": (models.inception_v3, 299),
#     "resnet50": (models.resnet50, 224),
#     "vgg16_bn": (models.vgg16_bn, 224),
# }
CLASSIFIERS = {
    "inception_v3": (models.inception_v3, 96),
    "resnet50": (models.resnet50, 224),
    "vgg16_bn": (models.vgg16_bn, 224),
}

#NUM_CLASSES = {
#    "imagenet": 1000
#}

NUM_CLASSES = {
    "imagenet": 40
}

# TODO: change the below to point to the ImageNet validation set,
# formatted for PyTorch ImageFolder
# Instructions for how to do this can be found at:
# https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
IMAGENET_PATH = "images/"
if IMAGENET_PATH == "":
    raise ValueError("Please fill out the path to ImageNet")

ch.set_default_tensor_type('torch.cuda.FloatTensor')

def norm(t):
    assert len(t.shape) == 4
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*ch.exp(lr*g)
    neg = (1-real_x)*ch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(x, g, lr):
    return x + lr*ch.sign(g)

def l2_prior_step(x, g, lr):
    new_x = x + lr*g/norm(g)
    norm_new_x = norm(new_x)
    norm_mask = (norm_new_x < 1.0).float()
    return new_x*norm_mask + (1-norm_mask)*new_x/norm_new_x

def gd_prior_step(x, g, lr):
    return x + lr*g

def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + ch.clamp(new_x - orig, -eps, eps)
    return proj

##
# Main functions
##

def make_adversarial_examples(image, true_label, args, model_to_fool, IMAGENET_SL):
    '''
    The main process for generating adversarial examples with priors.
    '''
    # Initial setup
    prior_size = IMAGENET_SL if not args.tiling else args.tile_size
    upsampler = Upsample(size=(IMAGENET_SL, IMAGENET_SL))
    total_queries = ch.zeros(args.batch_size)
    prior = ch.zeros(args.batch_size, 3, prior_size, prior_size)
    dim = prior.nelement()/args.batch_size
    prior_step = gd_prior_step if args.mode == 'l2' else eg_step
    image_step = l2_image_step if args.mode == 'l2' else linf_step
    proj_maker = l2_proj if args.mode == 'l2' else linf_proj
    proj_step = proj_maker(image, args.epsilon)
    print(image.max(), image.min())

    # Loss function
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    def normalized_eval(x):
        x_copy = x.clone()
        x_copy = ch.stack([F.normalize(x_copy[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
                        for i in range(args.batch_size)])
        return model_to_fool(x_copy)

    L = lambda x: criterion(normalized_eval(x), true_label)
    losses = L(image)

    # Original classifications
    orig_images = image.clone()
    orig_classes = model_to_fool(image).argmax(1).cuda()
    print("Original Class: " + str(orig_classes))
    print("True Label: " + str(true_label))
    print(true_label)
    correct_classified_mask = (orig_classes == true_label).float()
    total_ims = correct_classified_mask.sum()
    not_dones_mask = correct_classified_mask.clone()

    t = 0
    while not ch.any(total_queries > args.max_queries):
        t += args.gradient_iters*2
        if t >= args.max_queries:
            break
        if not args.nes:
            ## Updating the prior:
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = args.exploration*ch.randn_like(prior)/(dim**0.5)
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise)
            q2 = upsampler(prior - exp_noise)
            # Loss points for finite difference estimator
            l1 = L(image + args.fd_eta*q1/norm(q1)) # L(prior + c*noise)
            l2 = L(image + args.fd_eta*q2/norm(q2)) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(args.fd_eta*args.exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, args.online_lr)
        else:
            prior = ch.zeros_like(image)
            for _ in range(args.gradient_iters):
                exp_noise = ch.randn_like(image)/(dim**0.5)
                est_deriv = (L(image + args.fd_eta*exp_noise) - L(image - args.fd_eta*exp_noise))/args.fd_eta
                prior += est_deriv.view(-1, 1, 1, 1)*exp_noise

            # Preserve images that are already done,
            # Unless we are specifically measuring gradient estimation
            prior = prior*not_dones_mask.view(-1, 1, 1, 1)

        ## Update the image:
        # take a pgd step using the prior
        new_im = image_step(image, upsampler(prior*correct_classified_mask.view(-1, 1, 1, 1)), args.image_lr)
        image = proj_step(new_im)
        image = ch.clamp(image, 0, 1)
        l2_norm = 0.0
        if args.mode == 'l2':
            if not ch.all(norm(image - orig_images) <= args.epsilon + 1e-3):
                pdb.set_trace()
            else:
                l2_norm = norm(image - orig_images)
        else:
            if not (image - orig_images).max() <= args.epsilon + 1e-3:
                pdb.set_trace()
            else:
                l2_norm = (image - orig_images).max()

        ## Continue query count
        total_queries += 2*args.gradient_iters*not_dones_mask
        not_dones_mask = not_dones_mask*((normalized_eval(image).argmax(1) == true_label).float())

        ## Logging stuff
        new_losses = L(image)
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.sum()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
        not_done_loss = ((new_losses*not_dones_mask).sum()/not_dones_mask.sum()).cpu().item()
        max_curr_queries = total_queries.max().cpu().item()
        if args.log_progress:
            print("Queries: %d | Success rate: %f | Average queries: %f" % (max_curr_queries, current_success_rate, success_queries))

        if current_success_rate == 1.0:
            break

    return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'images_orig': orig_images.cpu().numpy(),
            'images_adv': image.cpu().numpy(),
            'all_queries': total_queries.cpu().numpy(),
            'correctly_classified': correct_classified_mask.cpu().numpy(),
            'success': success_mask.cpu().numpy(),
            'l2norm': l2_norm
    }

def main(args, model_to_fool, dataset_size):
    dataset = ImageFolder260(IMAGENET_PATH,
                    transforms.Compose([
                        transforms.Resize(dataset_size),
                        transforms.CenterCrop(dataset_size),
                        transforms.ToTensor(),
                    ]))
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size)
    total_correct, total_adv, total_queries = 0, 0, 0
    total_time, average_success_time, num_successes, avg_l2 = 0, 0, 0, 0.0
    #total_correct_classifications = 0
    for i, (images, targets) in enumerate(dataset_loader):
        print("Making Adversarial examples for batch " + str(i))
        #print(images)
        print(targets)

        timestart = time.time()
        if i*args.batch_size >= args.total_images:
            break
        res = make_adversarial_examples(images.cuda(), targets.cuda(), args, model_to_fool, dataset_size)
        ncc = res['num_correctly_classified'] # Number of correctly classified images (originally)
        num_adv = ncc * res['success_rate'] # Success rate was calculated as (# adv)/(# correct classified)
        queries = num_adv * res['average_queries'] # Average queries was calculated as (total queries for advs)/(# advs)
        timeend = time.time()

        time_to_run = (timeend - timestart)
        print("Num correct: " + str(ncc))
        print(" Num adversarial " + str(num_adv))
        total_correct += ncc
        total_adv += num_adv
        total_queries += queries

        orig = res['images_orig']
        img = res['images_adv']

        total_time += time_to_run
        if num_adv > 0:
            average_success_time += time_to_run
            num_successes += num_adv
            l2_norm = res['l2norm']
            #l2_norm = norm(img - orig)
            avg_l2 += l2_norm

        with open("report.txt", 'a') as f:
            f.write("*"*20)
            f.write("Image targets:\n ")
            f.write(str(targets))
            f.write("\nNum correctly classified by inception: " + str(ncc))
            f.write("\nSuccess rate of adversarial attacks: " + str(res['success_rate']))
            f.write("\nAverage number of queries: " + str(res['average_queries']))
            f.write("\nTotal time to run this query: " + str(time_to_run))
            if num_adv >0:
                f.write("\nL2 norm: " + str(l2_norm))
            f.write("*"*20)
            f.write("\n")

    with open("report.txt", 'a') as f:
        f.write("---"*20)
        f.write("\nFINAL REPORT: ")
        f.write("\nTotal time to run (includes failed classifications): " + str(total_time))
        f.write("\nAverage time to produce a successful attack: " + str(average_success_time/num_successes))
        f.write("\nAverage L2 norm: " + str(avg_l2/num_successes))
        f.write("\n Total number of correctly classified: " + str(total_correct))
        f.write("\n Total number of successful adversarial images: " + str(num_successes))

    # print("-"*80)
    # print("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
    #         aq=total_queries/total_adv,
    #         succ=total_adv/total_correct))
    # print("-"*80)

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params

    def __getattr__(self, x):
        return self.params[x.lower()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--log-progress', action='store_true')
    parser.add_argument('--nes', action='store_true')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--gradient-iters', type=int)
    parser.add_argument('--total-images', type=int)
    parser.add_argument('--classifier', type=str, default='inception_v3', choices=CLASSIFIERS.keys())
    args = parser.parse_args()

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = Parameters(defaults)
        args_dict = defaults

    model_type = CLASSIFIERS[args.classifier][0]
    model_to_fool = model_type(pretrained=True).cuda()
    model_to_fool = DataParallel(model_to_fool)
    model_to_fool.eval()

    with ch.no_grad():
        main(args, model_to_fool, CLASSIFIERS[args.classifier][1])
