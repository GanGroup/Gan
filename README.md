# **GAN for High-Resolution Animal Face Generation**

This project uses a **Generative Adversarial Network (GAN)** to generate high-quality, high-resolution images of animal faces. The model is trained on the **Animal Faces-HQ (AFHQ)** dataset and uses a **Deep Convolutional GAN (DCGAN)** architecture to create realistic images. The project explores different model architectures and data augmentation techniques to improve image quality.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Challenges and Solutions](#challenges-and-solutions)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run the Project](#how-to-run-the-project)
- [Future Work](#future-work)

## **Project Overview**
The primary goal of this project is to explore the use of GANs for generating realistic, high-resolution animal face images. The project involves:
- Building a **DCGAN** architecture with convolutional layers to capture spatial features in images.
- Utilizing data augmentation techniques to improve model generalization.
- Evaluating the model using the **Frechet Inception Distance (FID)** score, a popular metric for assessing the quality and diversity of generated images.

## **Dataset**
The model is trained on the **Animal Faces-HQ (AFHQ)** dataset, which contains 16,130 high-quality images of various animal faces at a resolution of 512x512 pixels. To address hardware limitations, images were resized to 128x128 pixels for this project.

## **Model Architecture**
The project uses a **Deep Convolutional GAN (DCGAN)** architecture, with separate architectures for the generator and discriminator:

- **Generator**:
  - Several convolutional transpose layers to upsample noise into a 128x128 image.
  - **Leaky ReLU** activations for non-linearity.
  - **Batch normalization** to stabilize training.

- **Discriminator**:
  - Convolutional layers to downsample the input image.
  - **Leaky ReLU** activations and **dropout** to prevent overfitting.

## **Training Process**
1. **Data Preprocessing**:
   - Resized all images to 128x128 pixels.
   - Applied data augmentation techniques (rotation, flipping) to increase diversity.

2. **Training**:
   - Trained the model for **4000 epochs**.
   - Used the **Adam optimizer** with learning rates of `0.0002` for both the generator and the discriminator.
   - Batch size: 64.

3. **Evaluation**:
   - The model's performance was evaluated using the **FID score**.
   - **Loss functions**: Binary cross-entropy loss for both the generator and discriminator.

## **Challenges and Solutions**
- **Mode Collapse**: 
  - Early in training, the generator produced repetitive outputs. 
  - **Solution**: Tuned the generator and discriminator balance, added more data, and applied **batch normalization** to stabilize training.

- **Data Augmentation Issues**:
  - Some augmentation techniques, like horizontal flipping, disrupted training.
  - **Solution**: Removed horizontal flipping and retained more beneficial augmentations, such as rotation.

## **Results**
- The model successfully generated realistic animal faces at a resolution of 128x128 pixels.
- **FID score** significantly improved from over 100 to below 50, indicating better image quality and diversity after training.
- Generated images showed significant diversity in animal face features and appearance.
![apply_new_dataset](https://github.com/user-attachments/assets/0d764986-d3a7-4aba-a3a3-e1691218b3ed)


## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - Deep Learning: **TensorFlow**, **Keras**
  - Data Processing: **Pandas**, **NumPy**
  - Image Augmentation: **TensorFlow ImageDataGenerator**
  - Visualization: **Matplotlib**
  
- **Evaluation Metrics**:
  - **Frechet Inception Distance (FID)**

## **Future Work**
- **Higher Resolution**: The current model was trained on 128x128 images. Future work will involve training on the full 512x512 resolution images to improve image quality.
- **Explore Advanced GANs**: Investigate more advanced GAN architectures such as **StyleGAN** or **BigGAN** to generate higher-quality images.
- **Different Datasets**: Apply the model to different datasets to explore broader applications of GANs in image generation.

---

### **Contact**
If you have any questions or suggestions, feel free to contact:
- **Author**: Kai Deng
- **Email**: kai.deng.job@outlook.com
