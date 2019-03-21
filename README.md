# ECE260FinalProject
An Evaluation of 3 different Black Box Adversarial attack

In this project, we evaluate:

ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models
Paper: https://arxiv.org/abs/1708.03999
Github: https://github.com/huanzhang12/ZOO-Attack

GenAttack: Practical Black-box Attacks with Gradient-Free Optimization
Paper: https://arxiv.org/abs/1805.11090
Github: https://github.com/nesl/adversarial_genattack

Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors
Paper: https://arxiv.org/abs/1807.07978
Github: https://github.com/MadryLab/blackbox-bandits

Changes have been made to the files to utilize the subset of Imagenet that we have made ( /images/ ).  
This subset consists of 40 images that are 500x3** pixels, and are named by this scheme: imagenet_id.class_name.image_number.jpg
The mapping between imagenet_id and the class name is found at https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.
Be aware that we have added 1 to the id, because the attack implementations use this added value, not the original.


