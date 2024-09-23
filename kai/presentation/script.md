## Start

Good afternoon, everyone. My supervisor is Dr Serhiy Yanchuk and This is kai. My presentation is about generating images using a deep Convolutional GAN. In this presentation, I am excited to take you through the fascinating world of GAN.



## Overview

This presentation is structured into five key sections: Introduction, Historical Models, Theoretical Background, Results, and Discussion.

## Introduction

I will give a brief overview of GANs and the main objectives of my thesis.

GAN is a concept introduced by Goodfellow in 2014. 

One of the most exciting applications of GAN is in image generation. GAN can generate realistic images in various domains.

However, training GAN is not without challenges. Problems like mode collapse, vanishing gradients and instability can cause model training to fail.

The objective of this thesis is to find the differences between GAN with dense layers and convolutional layers, explore the impact of layer depth and data augmentation, and finally, apply the GAN model to the Animal Faces-HQ dataset.



## Historical Models

In this part, I will introduce 3 milestones in image-generating history: Noise Contrastive Estimation NCE, Variational Autoencoder VAE and Diffusion model.

### Noise Contrastive Estimation (NCE)

Noise Contrastive Estimation NCE was introduced in 2010 by Gutmann as a way to estimate probability distributions by contrasting real data against noise.

Here, we can see the architecture of NCE. It begins with the input layer that receives both real data and noise. The data is then processed through a hidden layer, ultimately producing a binary output that indicates whether the input is real or noise.

Overhere is the architecture of NCE. The input real data and noise and then pass a hidden layer. the output is a binary shows whether the input is real or noise.

Its main applications include density estimation and language modelling, and it has laid a foundation for image and text generation methods.



NCE通常不会直接生成图片，但可以用作生成式模型中的损失优化方法，从而改进生成器的表现。

### Variational Autoencoder

Variational Autoencoder VAE, was introduced in 2013 by Kingma as a way to generate new data by encoding the input into a latent space and then reconstructing it. 



Here is the architecture of VAE, the left side is the encoder, which accepts the image and outputs a small latent space.

The right side is the decoder which upsamples the latent space to a new image.

VAEs have applications in image generation, data compression, and outlier detection, making them widely used in generative modelling.

Only 1 year later, GAN was introduced by Goodfellow. Over here we can see the architecture of VAE and GAN, very similar architecture. 

Just add a purple circle, and the new architecture can generate amazing, realistic images. I will show you why GAN can make it, but let's move to the last model first. 



使用训练好的解码器将潜在向量  z  作为输入，输出生成图片。

### Diffusion Model

Diffusion model was introduced in 2020 by Jonathan as a way to generate data by progressively adding noise to it and then learning to reverse the noise addition process. 

This two-step process involves adding noise during the forward phase and removing it during the reverse phase, which allows the model to generate high-quality images. 

Diffusion models have gained popularity for applications in image generation, audio synthesis, and even medical imaging.



## Theoretical Background

In this section, I will explain what is the objective function of GAN, describe the training process, and finally show you how to evaluate its performance.

### GAN Objective Function



The objective function is like a min-max game between two players. Think of the generator as an artist trying to create fake data that looks real, and the discriminator as a judge who tries to distinguish between real and fake data. Over time, the generator learns to produce data that’s harder and harder to distinguish from the real thing. This constant back-and-forth helps both models improve, pushing the generator to create highly realistic data.

Over here, we can see the min-max function of GAN.

p_data(x) is a probability distribution function of real data x.

x ~ p_data(x) a sample x drawn from p_data(x).



![image-20240919201519857](/Users/dengkai/Library/Application Support/typora-user-images/image-20240919201519857.png)

### Loss Function of GAN

The loss function of GAN has two parts: discriminator loss and generator loss.

![image-20240915085656550](/Users/dengkai/Library/Application Support/typora-user-images/image-20240915085656550.png)



This discriminator loss measures how well the discriminator predicts real data as real and generated data as fake. The better it performs, the lower the loss.

This generator loss measures how good the generator is at making the discriminator believe that the generated data is real.

![image-20240915085730854](/Users/dengkai/Library/Application Support/typora-user-images/image-20240915085730854.png)

And here is the loss function of the generator, it is not good. look at here: the blue curve is for function log(1-D), it shows in the early stage of training, the gradient is close to 0. This will cause the gradient vanishing problem, so we use log D to replace the original one to avoid the gradient vanishing issue.

x-axis y-axis

Horizontal and vertical



Here, we can see the function curve log(1-D) and log D. In the early stages of training, this function can lead to gradient vanishing. To address this, we modify the loss function to log D to avoid this issue.



加负号是为两grediant desent, 和 原函数对比是不用考虑它。当判别器给的概率很低时蓝线的斜率接近于0.

![image-20240915095834749](/Users/dengkai/Library/Application Support/typora-user-images/image-20240915095834749.png)

### GAN Training Process

先解释 黑，绿，蓝三条线代表的意义，然后从a到d讲绿线在训练过程中慢慢的向黑线靠近直至堆叠在黑线上

然后蓝线慢慢的向中间移动，并且变得水平。

蓝线的左边和右边都逐渐向中间移动，到达50%的水平，表明在模型训练完成后判别器不能很准确的判断真实图片和虚假的图片。



然后再解释z 和 x， 表示z 到x 是经有生成器进行映射的结果。



Here is the diagram of the GAN training process in distribution angle. 

先说出图总 z, x, 黑线，绿线和蓝线代表的意义

Initially, the generator produces poor-quality data, but over time, as the generator improves, the quality of the generated data becomes more similar to the real data. 



The key to training GAN is achieving a balance between the generator and the discriminator. This is shown in the distribution diagram, where the error gradually decreases over time, and the generated data distribution eventually aligns with the real data distribution.

![image-20240915085837801](/Users/dengkai/Library/Application Support/typora-user-images/image-20240915085837801.png)

蓝线代表的是判别器判别图片属于真实数据还是假数据的概率， 越高代表判别器认为该数据为真的概率越高。

当黑线高于绿线，判别器认为数据是真实数据，蓝线位置偏高 接近于1。

当黑线低于绿线， 如a 图过了交点后黑线低于绿线，判别器认为数据是假数据，蓝线位置接近于0.

**未达到平衡**：在a图中，生成器和判别器都还没有经过足够的训练，因此两者之间的“对抗”还处于不稳定状态。生成器生成的样本与真实数据之间的差异较大，因此判别器能够快速地判断真假，这使得输出值波动更大，表现为蓝线的上下起伏。



z uniform distribution, green curve normal distribution

x is sample space; it has the sample from real distribution and fake distribution 

a图中z映射到右边，所有绿线的峰值在右边 peak

蓝线高低起伏的原因是局部极小值导致的

绿线和黑线是概率密度函数 pdf， 描述的是数据点在不同位置出现的概率，越高的区域表示在该位置出现的概率越大。

### Optimal Discriminator

In this section, I will show you how to get the optimal discriminator.

Over here, we can see the objective function. V is a function of D and G, not a good format for finding the optimal D.

So we can use x to replace G(z), now V is a function of D.

to get the max value of D, we can convert it to an antiderivative function f(D(x)).

deriving and then setting the derivative to zero, we can get the optimal discriminator.

D(x) is a function of p_data (x) and p_g (x), it has 3 conditions.

when p_data(x) = p_g(x), the output of discriminator is 50%, 

### Evaluating GAN Performance



In most machine learning models, accuracy is a key evaluation metric. However, in GAN training 100% accuracy indicates the balance between the generator and the discriminator is broken.

On the right, the training log shows that I got a model with 100% accuracy, but when I use it to gen images, the quality is poor.

In GAN, we use FID to evaluate the model performance. 

0-10 perfect.

10-50 good. 

above 50 poor.



Since it can measure how similar the distribution of generated images is to the distribution of real images.

协方差矩阵的元素代表不同图像特征之间的线性关系，特别是这些特征如何协同变化。如果两个特征总是一起增加或减少，它们的协方差将是正的；如果一个增加而另一个减少，则协方差为负。协方差矩阵通过这些成对特征的协方差值来描述数据的分布结构。

## Results

In this part, I will show the experiment results for my objectives.

### Model Select

Before the experiment, I need to select a model.

In the early stages of my thesis, I focused on studying and implementing different types of GAN models. 

For example, week 1 is standard GAN. ....

In the second stage, I choose standard GAN for further study and here are the reasons.

\- **Foundational Model**: Obtaining standard GAN can help me learn other GAN models fast.

\- **Well-Documented**: I can easily find a document to help me solve the problem during the GAN training.

\- **Training Efficiency**: The architecture of Standard GAN is simple, and does not need a powerful GPU.

\- **Flexibility**: Easy to find a dataset to train and no need for extra work.

### Convolutional or Dense

The first is convolutional or Dense, in this experiment, I compared two main architectures: one using convolutional layers and the other using dense layers. The results showed that convolutional layers perform better.

dense 8 ms

convolution 20 ms

### Exploring Layer Depth in GAN



The second is layer depth. in this experiment, I explored the different layer depths—3, 4, 5, and 6 layers trained them 3 times and evaluated the results using the average FID score.

And over here is the result table. It shows if I singly increase the layers in the generator or discriminator, the model performance will get worse. But if I keep the layer balance, the model performance will get better.





### Impact of Data Augmentation

Then, I explored the 3 data augmentation techniques, including rotation, shifting and flipping, and trained them with the Mnist dataset 3 times. Evaluated the results using the average FID score.

Over here is the result table. It shows the GAN model without data augmentation techniques performs better.

in this scenario, I assume the data augmentation will disturb the real data distribution and make GAN model performance worse.

### Applying Animal Faces HQ Dataset

Finally, I apply the standard GAN model with the animal faces HQ dataset.

The data set has about 16 thousand images, each 512*512 pixels.

指着架构图讲

The noise is a 100-dimensional vector.

The generator has 5 transpose convolutional layers with leaky relu active function.

The discriminator also has 5 convolutional layers with leaky relu active function.

the optimiser is Adam, the training rate is 0.002, trained with 4000 epochs.

Here are the cat face I genes; the cat is realistic and high quality.

## Discussion

In this section, I will show you a brief summary and the future work for my objectives.

### Summary and future work

1. The first is dense or Convolutional layers. The result shows the convolutional layers structure performs better.
2. The second is exploring layer dept. The result table shows it is significant for us to keep the balance between the generator and discriminator when designing the structure of GAN. In future, I will explore the probability of more depth structure with u net and res net.
3. The next is the impact of data augmentation. Based on the result, I assume that data augmentation may disturb the real data distribution, but it is not sufficient, in the future I will test all the data augmentation techniques to support my assumption.
4. Finally, I applied the GAN model with the Animal Faces-HQ dataset. The result shows it can gen realistic cat faces, but some of them are blurred, in the future, I will extend the training dataset and train the model for more epochs.

