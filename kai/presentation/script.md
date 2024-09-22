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

### Mathematical Formulation during GAN Training 1

Let's prove it.





In this section, we’ll go over the mathematical formulation that governs GAN training.

First, the problem setup: The goal of GANs is to train two networks—a generator, denoted as G , and a discriminator, denoted as D —so that the generator learns to produce data that closely resembles real data. The discriminator’s role is to differentiate between real and generated data. The overall objective function used in this process is represented as:



Show the first equation on the slide

This function is essentially the sum of two terms. The first term measures how well the discriminator classifies real data, while the second term measures how well it distinguishes between real and generated data.



Next, we rewrite this objective function in integral form:

Show the rewritten objective function

Here, x' represents the data generated by the generator, which we call G(z) . This form allows us to reflect the contribution of generated samples more explicitly.



Finally, the objective function simplifies to the following expression:

Show the final equation

This is the fundamental equation that guides the adversarial training process between the generator and discriminator, where the goal is to maximize the likelihood of real data while minimizing the likelihood of generated data being detected as fake.

### Mathematical Formulation during GAN Training 2



Simplifying to a Basic Function

In this part, we’re simplifying the objective function from its integral form into a more manageable equation, essentially a basic function. The formula on the slide represents this function, where f(D(x)) shows how real data p_{data}(x) and generated data p_g(x) interact in the model’s training process.

Deriving the Optimal Discriminator

To determine the best possible discriminator, we need to take the derivative of the objective function with respect to D(x) . This process allows us to calculate how much D(x) changes based on the values of real and generated data probabilities.”

Setting the Derivative to Zero

Once we’ve taken the derivative, the next step is to set it equal to zero. This leads us to the optimal formula for the discriminator, where D^*(x) is expressed as a ratio between the real and generated data probabilities. In simpler terms, this formula helps the model differentiate between real and generated data as accurately as possible.



### Verifying the Optimal Discriminator

 • When p_data (x) is larger than p_g (x) ,D^∗ (x) ≈ 1, indicating that the data point is almost certainly from the real data.

  • When p_data (x) is much smaller than p_g (x), D^∗ (x) ≈ 0, indicating that the data point is almost certainly from the generated data.

  • When p_data (x) is close to p_g (x), D^∗ (x) ≈ 0.5, indicating that the discriminator cannot confidently determine whether the data point is realor generated, giving each a 50% probability.

### Evaluating GAN Performance



In most machine learning models, accuracy is a key evaluation metric. However, in the case of GAN training, achieving 100% accuracy indicates a problem. It means that the balance between the generator and the discriminator is broken.

For example, I once saw a training log showing that the discriminator had 100% accuracy, and I thought I had built the perfect GAN model. However, the images generated were poor. After further investigation, I realized the issue.

FID measures how similar the distribution of generated images is to the distribution of real images.



协方差矩阵的元素代表不同图像特征之间的线性关系，特别是这些特征如何协同变化。如果两个特征总是一起增加或减少，它们的协方差将是正的；如果一个增加而另一个减少，则协方差为负。协方差矩阵通过这些成对特征的协方差值来描述数据的分布结构。

## Results



### Model Select



In the early stages of my thesis, I focused on implementing and studying different types of GAN models. As shown here, I worked through a range of GAN variations, from the standard GAN to more advanced architectures like conditional GANs, style GANs, and even domain transfer GANs. Each notebook helped me explore the unique aspects of these models and understand their strengths and limitations.

After gaining hands-on experience with these variants, I decided to proceed with the **Standard GAN** for several reasons:

\- **Foundational Model**: The standard GAN serves as the base for nearly all other GAN models, providing a strong starting point.

\- **Well-Documented**: It’s a well-studied model, with extensive research available, which makes it easier to troubleshoot and refine.

\- **Training Efficiency**: It’s computationally less demanding compared to more complex models, making it feasible to train with limited hardware resources.

\- **Flexibility**: The standard GAN is adaptable to various image generation tasks, making it a solid baseline for comparison with more advanced models.

Working through these different models not only provided foundational knowledge but also informed my decision to use the Standard GAN for this project.



### Convolutional or Dense



I compared two main architectures: one using convolutional layers and the other using dense layers. Our results showed that convolutional layers significantly outperformed dense layers in generating high-quality images.

dense 8 ms

convolution 20 ms



### Exploring Layer Depth in GAN



In this slide, I explore how increasing the layer depth in both the generator and discriminator affects GAN performance. I experimented with different layer depths—3, 4, 5, and 6 layers—and evaluated the results using the FID score.



My findings show that adding layers to only one network leads to performance degradation, with unstable training and worse image quality. However, when both the generator and discriminator have balanced, increased depth, I observed better results, with lower FID scores and higher-quality images.



In conclusion, balancing the depth of both networks is crucial to achieving optimal GAN performance.





### Impact of Data Augmentation 1



Let’s first talk about the objective of this part of the study. The goal was to evaluate the impact of different data augmentation techniques on the performance of GAN. We specifically tested three main augmentation techniques: rotating the images by 10 degrees, shifting the images both vertically and horizontally by 0.1, and flipping the images horizontally.



Now, here’s the interesting part. The results showed that, surprisingly, the model without any data augmentation performed better than when these techniques were applied. This suggests that the augmentation methods, such as flipping or rotating, introduced more noise into the system rather than improving its performance. As a result, certain augmentation techniques like flipping and rotation were removed to optimize the model.



### Impact of Data Augmentation 2



Now, let’s take a closer look at the results. As we can see from the table, the model that used no data augmentation at all achieved the best FID score of 58.94, indicating relatively better image quality. On the other hand, models that used a combination of techniques such as rotation, shifting, and flipping resulted in much higher FID scores—over 100 in some cases—indicating poorer image quality.



The assumption here is that data augmentation techniques like rotation and flipping disturbed the real data distribution, introducing noise rather than enhancing the learning process. The FID score represents how well the generated images resemble real images, and lower scores are better. The ideal FID score is below 50, with anything over that considered poor quality.





### Applying Animal Faces HQ Dataset 1



Let's begin by introducing the dataset used for our model. The **Animal Faces-HQ (AFHQ)** dataset consists of **16,130 high-quality images** of animal faces, including various species such as cats, dogs, and wildlife. These images are originally **512x512 pixels** but have been **downscaled to 128x128 pixels** to optimize memory usage during training. This allows for faster processing without compromising too much on the quality of the generated images.





**Model Structure:**

 • The generator takes a 100-dimensional noise vector, has 5 transpose convolutional layers with LeakyReLU activation and batch normalization, and outputs a 3×128×128 image.

 • The discriminator takes a 3×128×128 image as input, has 5 convolutional layers with LeakyReLU activation, batch normalization, and dropout and outputs a single probability value indicating real or fake data.

**Training Details:**

 • Optimizer: Adam.

 • Learning rate: 0.0002.

 • Epochs: 4000 epochs of training using this dataset.



### Applying Animal Faces HQ Dataset 2



Next, let's discuss the model structure used in this experiment. 

\- **The generator** takes a **100-dimensional noise vector** and consists of **five transpose convolutional layers**. Each layer uses **LeakyReLU activation**, with **batch normalization** applied to stabilize the training process. The generator then outputs an image of size **3x128x128**.

\- **The discriminator**, on the other hand, accepts a **3x128x128 image** as input. It consists of **five convolutional layers** with **LeakyReLU activation**, **batch normalization**, and **dropout layers** to prevent overfitting. The output of the discriminator is a **single probability value** that determines whether the image is real or fake.

For training, we used the **Adam optimizer** with a **learning rate of 0.0002** and trained the model for **4000 epochs** using this dataset.



### Applying Animal Faces HQ Dataset 4



Here, you can see the results of our model applied to the **Animal Faces-HQ dataset**. The images displayed here are all generated by our model after training. Notice how the GAN has successfully learned to generate animal faces that closely resemble the original dataset. While there are some imperfections in a few images, the overall quality of these generated faces is impressive, considering the complexity of the task and the amount of data used for training.





## Discussion



### Key Findings



When it comes to **model performance**, the generator showed the ability to produce images that closely resembled real data. However, there’s still room for improvement in generating consistently high-quality outputs. Similarly, the **discriminator** performed effectively in differentiating between real and generated images, but like with any adversarial model, this could be further refined.



Several **challenges** emerged during the process. For instance, using data augmentation—while intended to diversify training—introduced noise, which led to a slight decline in performance. Additionally, we observed some **training instability**, which is typical in adversarial setups, where the generator and discriminator often struggle to balance their progress.



Finally, regarding **evaluation**, the model achieved more consistent results without data augmentation. The FID scores suggest that for this particular dataset and model architecture, data augmentation may have added unnecessary complexity, impacting the overall outcome.





### Future Work



First, we could focus on **model improvements**. By tweaking the **hyperparameters**, particularly in the training process, we can address some of the stability issues observed. Furthermore, exploring more advanced architectures, such as **StyleGAN**, could offer improved image generation quality.



As for the **dataset**, using a larger and more diverse set of images will help the model generalize better, potentially enhancing the performance on unseen data. The **Animal Faces HQ** dataset was a good starting point, but a more varied dataset could provide better insights into the model’s capabilities.



Lastly, we would like to experiment with **adaptive data augmentation methods**. Rather than applying fixed augmentation like flipping or rotating, we can explore techniques that adapt to the model’s current performance, potentially enhancing stability without introducing noise.