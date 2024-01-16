# ResNet on CIFAR-10

<hr>

## Contents

1. [Highlights](#Highlights)
2. [ResNet Primer](#ResNet)
3. [Requirements](#Requirements)
4. [Usage](#Usage)
5. [Results](#Results)


<hr>

## Highlights
This project is a implementation from scratch of a slightly modified version of the vanilla ResNet introduced in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). We implement this model on the small scale benchmark dataset `CIFAR-10`. One of the goals of this project is to illustrate the speed gain of the ResNet model in comparison to the vision transformer models while maintaining comparable accuracy on `CIFAR-10` and other small-scale datasets. 

For an extremely fast ResNet model on `CIFAR-10`, check out David Page's amazing blog post [How to Train Your ResNet 8: Bag of Tricks](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/), where a modified ResNet is trained to reach 94% accuracy in 26 seconds on a V100 GPU.

<hr>

## ResNet Primer
ResNet, short for Residual Network is a specific type of neural network that was introduced in 2015 by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun in their paper “Deep Residual Learning for Image Recognition”. The ResNet models were extremely successful which you can see from the following results:
* Won 1st place in the ILSVRC 2015 classification competition with a top-5 error rate of
      3.57% (An ensemble model).
* Won the 1st place in ILSVRC and COCO 2015 competition in ImageNet Detection,
      ImageNet localization, Coco detection and Coco segmentation.  
* Replacing VGG-16 layers in Faster R-CNN with ResNet-101. They observed relative
      improvements of 28%.
* Efficiently trained networks with 100 layers and 1000 layers also.  

In order to solve complex problems, we stack some additional layers in the Deep Neural Networks which results in improved accuracy and performance. The intuition behind adding more layers is that these layers progressively learn more complex features. For example, in case of classifying images, the first layer may learn to detect edges, the second layer may learn to identify textures and similarly the third layer can learn to detect objects and so on. But it has been found that there is a maximum threshold for depth with the traditional Convolutional neural network model as a result of the vanishing gradient problem.

 How can we build deeper networks? Enter the residual block:

<img src="./images/res1.png" width="550"></img>

The main component of the residual block is the skip connection.  Skip connections in ResNet architectures provide a direct link between earlier and later layers, facilitating information preservation and ease of training in deep networks. They enable the network to learn residual mappings rather than full transformations, addressing the vanishing gradient problem. This technique allows for the construction of much deeper neural networks without performance degradation.

A typical ResNet architecture looks like the following:

<img src="./images/res2.png" width="550"></img>

<hr>

## Requirements
```shell
pip install -r requirements.txt
```

<hr>

## Usage
To replicate the reported results, clone this repo
```shell
cd your_directory git clone git@github.com:jordandeklerk/ResNet-pytorch.git
```
and run the main training script
```shell
python train.py 
```
Make sure to adjust the checkpoint directory in train.py to store checkpoint files.

<hr>

## Results
We test our approach on the `CIFAR-10` dataset with the intention to extend our model to 4 other small low resolution datasets: `Tiny-Imagenet`, `CIFAR100`, `CINIC10` and `SVHN`. All training took place on a single A100 GPU.
  * CIFAR10
    * ```resnet_cifar10_input32``` - 90.7 @ 32

Flop analysis:
```
total flops: 584217600
total activations: 679946
number of parameter: 11200882
| module   | #parameters or shape   | #flops   |
|:---------|:-----------------------|:---------|
| model    | 11.201M                | 0.584G   |
|  0       |  28.768K               |  29.458M |
|  1       |  11.167M               |  0.555G  |
|  2       |  5.13K                 |  13.312K |
```
   
<hr>

## Citations
```bibtex
@article{He2015,
	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
	title = {Deep Residual Learning for Image Recognition},
	journal = {arXiv preprint arXiv:1512.03385},
	year = {2015}
}
```
