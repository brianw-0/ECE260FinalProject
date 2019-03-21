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


Each folder contains it's own algorithm and running instructions, but here are the commands that we have used.

For Blackbox Bandits:
```
python main.py --json-config=configs/nes-l2-final.json 
```
You can alter this json file to set specific parameters i.e. max number of iterations

For genAttack:
```
python main.py --input_dir=./images/ --test_size=40 --eps=1 --alpha=0.15 --mutation_rate=0.1 --max_steps=1500 --output_dir=genattack_run3 --resize_dim=96 --pop_size=6 --adaptive=True
```
Where output_dir is the folder name that you want to save your output to.  

For ZOO Attack:
```
python test_all.py -a black -d imagenet -n 40 --solver adam --untargeted --use_resize --init_size=96 -s "black_results"

```
Where "black_results" is the name of the output directory, and the attack should resize the images to 96x96 for higher efficiency.



It would be much easier if you set up a virtual environment to run this code.  This was tested with Python 3.5+.  You can use
```
pip install -r requirements.txt
```
to get most of the packages.  Some might be missing, just a warning


