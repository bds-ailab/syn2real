## General Approach

### Baseline classifier for problem demonstration


Although synthetic data offers significant advantages for training deep learning models on common tasks such as classification, tracking, and segmentation, thanks to its adaptability and complete control over image content, it also comes with certain limitations. The primary limitation of using synthetic data is that training a model solely on these datasets introduces an overfitting bias, leading to a drop in performance when testing the model on real-world images. This issue, commonly referred to as "Domain Shift." in the state of the art, highlights the gap between synthetic and real data domains. To demonstrate this phenomenon, we used the [Syn2real Benchmark](https://ai.bu.edu/syn2real/) dataset, which includes annotated images of synthetic and real objects, to train a simple ResNet50-based classifier and then analyzed its performance.

![image](https://github.com/user-attachments/assets/6838ab6d-b189-4aca-b2c0-d87ff1de6c87)

### Classification performance drop

When trained on synthetic images only, the model does not show strong generalization results on real images (target domain), resulting in a 50-60% drop in accuracy.

![image](https://github.com/user-attachments/assets/31e1877c-0bb9-4670-a356-383688c5691e)

### Synthetic objects variability bias

Using prediction explainability methods, we determined that the lack of diversity in synthetic images, such as in object colors and patterns, was one of the reasons for the performance drop when tested on real images.
![image](https://github.com/user-attachments/assets/e756f00c-7f44-4365-86e9-3d0bf137437a)

To further visualize the domain shift problem, we applied UMAP projection on both source and target domains in the feature map space of our baseline model. We observed that the model successfully distinguished between the different classes during the training phase, as indicated by the disjoint clusters in the source domain. However, when evaluated on the real images, there was a shift of the embeddings towards the center of the projection plane due to the newly added features associated with realism, such as new colors, patterns, backgrounds, and noise.
![image](https://github.com/user-attachments/assets/1848786f-c451-4057-aa59-70c004589416)

### Domain shift on segmentation task

Similarly, we demonstrated this effect on segmentation tasks using the DeepLabv3 model, which we trained on synthetic GTA images and tested on real-world Cityscapes images. The segmentation performance was evaluated using mean Intersection over Union (IoU) across all classes. As shown, the large performance gap between training on synthetic data and testing on real images is due to the model overfitting on the textures and appearances of synthetic objects, which prevents it from generalizing effectively to real images.
![image](https://github.com/user-attachments/assets/5a928f0e-b1af-4e38-9242-a51a9eb652a7)

### Proposed solution

To reduce the performance drop between synthetic and real images, augmentation methods must be implemented to add realism and diversity to the training images. This helps to align the feature maps of the two domains for better generalization results. We propose a generative models-based approach to achieve this domain adaptation between synthetic and real domains. This approach involves transforming synthetic images to the real domain by augmenting their realism while preserving the original semantic layout and associated annotation maps.
For the generation task, we employ a controlled Diffusion Model (DM) guided by an input conditioning image that integrates information about the semantic segmentation of the synthetic image and the scene details. Using prompt inversion, we train the model in an unsupervised manner to alter the image style from the synthetic to the real domain. Adopting an active learning approach, we iteratively train the model to enhance the generation results.
Through qualitative and quantitative studies, we demonstrate that our model outperforms traditional domain adaptation methods in terms of enhancing realism and semantic alignment with real-world data. Additionally, we show that our model improves the performance of independent deep learning models when trained solely on the generated data and tested on real-world data. Furthermore, we illustrate that our model surpasses recent style transfer models, such as CycleGAN-Turbo, while requiring significantly less computing power.

**Note:** One advantage of our method is its generative nature. Our approach generates new augmented (realistic) instances that can be used for purposes beyond classification while reducing the difference with the target domain. This contrasts with other domain adaptation methods that only modify the model's internal feature map embeddings and thus need to be adapted for each specific task and model architecture.
