## Start

Good afternoon, everyone. My presentation is about generating images using a deep convolutional GAN. In this presentation, I am excited to take you through the fascinating world of GAN.



## Overview

This presentation is structured into five key sections: Introduction, Historical Models, Theoretical Background, Results, and Discussion.

In the Introduction, I will provide a brief overview of GAN and the objectives of my thesis.

Next, in the Historical Models section, I will introduce 3 significant image generation models in recent years.

In the Theoretical Background section, I will explain the key concepts behind GAN, focusing on the interaction between the generator and discriminator through adversarial training.

Following that, I will present the experimental results of our GAN training and evaluate the performance of different architectures.

Finally, in the Discussion, I will highlight the main findings and discuss potential directions for future work.



## Introduction



In this project, I’ll be exploring the world of Generative Adversarial Network or GAN, a concept introduced by Goodfellow back in 2014. 

One of the most exciting applications of GAN is in image generation. GAN has been used to generate realistic images in various domains, including art, medical imaging, and even face generation. Its versatility is one of the reasons it has gained so much popularity.

However, training GAN is not without challenges. Problems like mode collapse, where the generator produces limited variation and instability during training make it a difficult model to optimize. That’s why the objective of my study is to explore ways to enhance the training stability and improve the quality of the generated images.



## Historical Models

In this part, I will introduce 3 image generated models, Noise Contrastive Estimation NCE, Variational Autoencoder VAE and Diffusion model.

### Noise Contrastive Estimation (NCE)

Noise Contrastive Estimation NCE was introduced in 2010 by Gutmann as a way to estimate probability distributions by contrasting real data against noise. In its architecture, NCE uses real and noise samples as input and applies binary classification to learn effectively. Its main applications include density estimation and language modelling, and it has laid a foundation for image and text generation methods.

### Variational Autoencoder

Variational Autoencoders, or VAEs, were introduced in 2013 by Kingma as a way to generate new data by encoding the input into a latent space and then reconstructing it. VAEs consist of an encoder, which compresses the data, and a decoder, which reconstructs it. VAEs have applications in image generation, data compression, and anomaly detection, making them widely used in generative modelling.

### Diffusion Model

Diffusion models, introduced in 2020 by Jonathan, generate data by progressively adding noise to it and then learning to reverse the noise addition process. This two-step process involves adding noise during the forward phase and removing it during the reverse phase, which allows the model to generate high-quality images. Diffusion models have gained popularity for applications in image generation, audio synthesis, and even medical imaging.





## Theoretical Background

In this section, I will explain what is GAN, walk through its objective function, describe the training process, and show how to evaluate its performance.

### Generative Adversarial Network

Generative Adversarial Network, or GAN, was introduced by Ian Goodfellow in 2014. Simply put, GANs involve two models that work together—the generator and the discriminator. Think of the generator as an artist trying to create fake data that looks real, and the discriminator as a judge who tries to distinguish between real and fake data. Over time, the generator learns to produce data that’s harder and harder to distinguish from the real thing. This constant back-and-forth helps both models improve, pushing the generator to create highly realistic data.

Now, let’s take a look at how GANs are structured.

In terms of structure, GAN has two core parts: the generator and the discriminator. The generator starts with random noise and attempts to turn it into data that looks real. Meanwhile, the discriminator takes both real data and the generator’s fake data and tries to tell which is which. The generator is essentially trying to trick the discriminator, while the discriminator is trying to become an expert in spotting fake data. This process, called adversarial training, pushes the generator to constantly improve its creations until they’re nearly indistinguishable from the real data.

Now, let’s explore some of the practical applications of GANs.

GAN has made a big impact across several fields, especially powerful in generating realistic images and videos, which can be used for things like creating synthetic images for training other models or even for creative purposes like art. It also has applications in image restoration—think of restoring old or damaged photos to their former glory. In addition, GAN is used in style transfer, which allows us to blend the style of one image with the content of another. They’ve even been used to generate completely lifelike human faces, which can be useful in video games, movies, and virtual environments.

### GAN Objective Function

In this section, let’s break down the core objective function of GANs.

The entire system operates like a game between two players—the generator and the discriminator. The goal for the generator is to create data that is so realistic, the discriminator cannot tell it’s fake. Meanwhile, the discriminator’s job is to become an expert at distinguishing between real and generated data.

Formally, this is captured in what we call a **minimax game**. The generator is trying to minimize the likelihood that the discriminator identifies its generated data as fake, while the discriminator is trying to maximize the difference between real and generated data. In simpler terms, the generator is always trying to “fool” the discriminator, while the discriminator is always getting better at spotting the fakes.

The generator’s objective is to reduce the probability that the discriminator can correctly identify fake data. It wants the discriminator to believe the generated data is real. On the other hand, the discriminator’s objective is to correctly tell real from fake. It’s a constant push-and-pull between the two networks, which results in both improving over time. This dynamic helps GANs produce incredibly realistic images, videos, and more.



### Loss Function of GAN



Starting with the discriminator’s loss function—its role is to classify data as real or fake. The discriminator’s loss function calculates how well it distinguishes between the real data it’s shown and the fake data generated by the generator. Mathematically, it’s represented as:

![image-20240915085656550](/Users/dengkai/Library/Application Support/typora-user-images/image-20240915085656550.png)



This function measures how well the discriminator predicts real data as real and generated data as fake. The better it performs, the lower the loss.

Next, we have the generator’s loss function. The generator aims to generate data that can “fool” the discriminator into thinking it’s real. Its loss function is simpler:

![image-20240915085730854](/Users/dengkai/Library/Application Support/typora-user-images/image-20240915085730854.png)



This function measures how good the generator is at making the discriminator believe that the generated data is real.

In short, the generator tries to minimize its loss by creating more convincing data, while the discriminator tries to minimize its own loss by getting better at spotting fake data. This adversarial process drives both networks to improve, leading to increasingly realistic results.



### GAN Training Process

The training process of GAN is adversarial. Initially, the generator produces poor-quality data, but over time, as the generator improves, the quality of the generated data becomes more similar to the real data. The key to training GANs is achieving a balance between the generator and the discriminator. This is shown in the distribution diagram, where the error gradually decreases over time, and the generated data distribution eventually aligns with the real data distribution.

![image-20240915085837801](/Users/dengkai/Library/Application Support/typora-user-images/image-20240915085837801.png)

蓝线代表的是判别器判别图片属于真实数据还是假数据的概率， 越高代表判别器认为该数据为真的概率越高。

当黑线高于绿线，判别器认为数据是真实数据，蓝线位置偏高 接近于1。

当黑线低于绿线， 如a 图过了交点后黑线低于绿线，判别器认为数据是假数据，蓝线位置接近于0.



z uniform distribution, green curve normal distribution

a图中z映射到右边，所有绿线的峰值在右边 peak

蓝线高低起伏的原因是局部极小值导致的

绿线和黑线是概率密度函数 pdf， 描述的是数据点在不同位置出现的概率，越高的区域表示在该位置出现的概率越大。

### Mathematical Formulation during GAN Training 1

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



### Evaluating GAN Performance



In most machine learning models, accuracy is a key evaluation metric. However, in the case of GAN training, achieving 100% accuracy indicates a problem. It means that the balance between the generator and the discriminator is broken.



For example, I once saw a training log showing that the discriminator had 100% accuracy, and I thought I had built the perfect GAN model. However, the images generated were poor. After further investigation, I realized the issue.



If the discriminator is too strong and consistently identifies images as fake (with 100% accuracy), the generator doesn’t receive any positive feedback and can’t learn what it needs to generate, leading to failed training.



To illustrate, imagine trying to generate tiger images. The generator first creates a lion, then a cat, then a dog, and each time the discriminator says “No, this is not a tiger.” After multiple attempts, the generator never learns what a tiger looks like.



On the other hand, if both the generator and discriminator are weak, the generator may generate something incorrect, like a tiger-skin cake, but the weak discriminator says it’s a tiger. Through this flawed feedback, the generator learns incorrect features and ultimately produces poor images like a cat with tiger skin, which still passes as a tiger.

## Results



### Model Select



In the early stages of my project, I focused on implementing and studying different types of GAN models. As shown here, I worked through a range of GAN variations, from the standard GAN to more advanced architectures like conditional GANs, style GANs, and even domain transfer GANs. Each notebook helped me explore the unique aspects of these models and understand their strengths and limitations.

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