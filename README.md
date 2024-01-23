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
This project is an implementation from scratch of a slightly modified version of the vanilla ResNet introduced in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). We implement this model on the small scale benchmark dataset `CIFAR-10`. One of the goals of this project is to illustrate the speed gain of the ResNet model in comparison to the vision transformer models while maintaining comparable accuracy on `CIFAR-10` and other small-scale datasets. 

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

Model summary:
```
=============================================================================================
Layer (type:depth-idx)                        Kernel Shape     Output Shape     Param #
=============================================================================================
ResNet                                        --               [1, 10]          --
├─Stem: 1-1                                   --               [1, 64, 32, 32]  --
│    └─ConvBlock: 2-1                         --               [1, 32, 32, 32]  --
│    │    └─Conv2d: 3-1                       [3, 3]           [1, 32, 32, 32]  864
│    │    └─BatchNorm2d: 3-2                  --               [1, 32, 32, 32]  64
│    │    └─SiLU: 3-3                         --               [1, 32, 32, 32]  --
│    └─ConvBlock: 2-2                         --               [1, 32, 32, 32]  --
│    │    └─Conv2d: 3-4                       [3, 3]           [1, 32, 32, 32]  9,216
│    │    └─BatchNorm2d: 3-5                  --               [1, 32, 32, 32]  64
│    │    └─SiLU: 3-6                         --               [1, 32, 32, 32]  --
│    └─ConvBlock: 2-3                         --               [1, 64, 32, 32]  --
│    │    └─Conv2d: 3-7                       [3, 3]           [1, 64, 32, 32]  18,432
│    │    └─BatchNorm2d: 3-8                  --               [1, 64, 32, 32]  128
│    │    └─SiLU: 3-9                         --               [1, 64, 32, 32]  --
├─ResidualStack: 1-2                          --               [1, 512, 4, 4]   --
│    └─ResidualBlock: 2-4                     --               [1, 64, 32, 32]  1
│    │    └─Identity: 3-10                    --               [1, 64, 32, 32]  --
│    │    └─BasicResidual: 3-11               --               [1, 64, 32, 32]  73,984
│    │    └─SiLU: 3-12                        --               [1, 64, 32, 32]  --
│    └─ResidualBlock: 2-5                     --               [1, 64, 32, 32]  1
│    │    └─Identity: 3-13                    --               [1, 64, 32, 32]  --
│    │    └─BasicResidual: 3-14               --               [1, 64, 32, 32]  73,984
│    │    └─SiLU: 3-15                        --               [1, 64, 32, 32]  --
│    └─MaxPool2d: 2-6                         2                [1, 64, 16, 16]  --
│    └─ResidualBlock: 2-7                     --               [1, 128, 16, 16] 1
│    │    └─ConvBlock: 3-16                   --               [1, 128, 16, 16] 8,448
│    │    └─BasicResidual: 3-17               --               [1, 128, 16, 16] 221,696
│    │    └─SiLU: 3-18                        --               [1, 128, 16, 16] --
│    └─ResidualBlock: 2-8                     --               [1, 128, 16, 16] 1
│    │    └─Identity: 3-19                    --               [1, 128, 16, 16] --
│    │    └─BasicResidual: 3-20               --               [1, 128, 16, 16] 295,424
│    │    └─SiLU: 3-21                        --               [1, 128, 16, 16] --
│    └─MaxPool2d: 2-9                         2                [1, 128, 8, 8]   --
│    └─ResidualBlock: 2-10                    --               [1, 256, 8, 8]   1
│    │    └─ConvBlock: 3-22                   --               [1, 256, 8, 8]   33,280
│    │    └─BasicResidual: 3-23               --               [1, 256, 8, 8]   885,760
│    │    └─SiLU: 3-24                        --               [1, 256, 8, 8]   --
│    └─ResidualBlock: 2-11                    --               [1, 256, 8, 8]   1
│    │    └─Identity: 3-25                    --               [1, 256, 8, 8]   --
│    │    └─BasicResidual: 3-26               --               [1, 256, 8, 8]   1,180,672
│    │    └─SiLU: 3-27                        --               [1, 256, 8, 8]   --
│    └─MaxPool2d: 2-12                        2                [1, 256, 4, 4]   --
│    └─ResidualBlock: 2-13                    --               [1, 512, 4, 4]   1
│    │    └─ConvBlock: 3-28                   --               [1, 512, 4, 4]   132,096
│    │    └─BasicResidual: 3-29               --               [1, 512, 4, 4]   3,540,992
│    │    └─SiLU: 3-30                        --               [1, 512, 4, 4]   --
│    └─ResidualBlock: 2-14                    --               [1, 512, 4, 4]   1
│    │    └─Identity: 3-31                    --               [1, 512, 4, 4]   --
│    │    └─BasicResidual: 3-32               --               [1, 512, 4, 4]   4,720,640
│    │    └─SiLU: 3-33                        --               [1, 512, 4, 4]   --
├─Head: 1-3                                   --               [1, 10]          --
│    └─AdaptiveAvgPool2d: 2-15                --               [1, 512, 1, 1]   --
│    └─Flatten: 2-16                          --               [1, 512]         --
│    └─Dropout: 2-17                          --               [1, 512]         --
│    └─Linear: 2-18                           --               [1, 10]          5,130
=============================================================================================
Total params: 11,200,882
Trainable params: 11,200,882
Non-trainable params: 0
Total mult-adds (M): 582.86
=============================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 10.88
Params size (MB): 44.80
Estimated Total Size (MB): 55.69
=============================================================================================
```

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
