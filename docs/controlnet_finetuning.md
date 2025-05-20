## ControlNet FineTuning ## 
<br>

After conducting a sanity check by training ControlNet on a symbolic dataset of circles, our model is now prepared for fine-tuning on actual synthetic/real images. In these initial experiments, we will only train the ControlNet component of our general strategy, excluding the discriminator block. The goal is to ensure that the ControlNet block can distinguish between the two domains and is capable of converting images from the synthetic domain to the real domain. As expected, we anticipate generations with artifacts, which typically occur when attempting to control the diffusion process. However, we plan to address this issue with a second discriminator block that will refine the generation through adversarial training.

For training, we will utilize a dataset that resembles synthetic images from the GTA game and real images from the Cityscapes dataset. This dataset's advantage lies in the segmentation labels for each image, which are consistent across both the synthetic and real domains, sharing the same class IDs and color codes. To train the model, we captioned each image using the BLIP model and then appended the tokens 'a real picture of'/'a synthetic picture of' to guide the model's output generation. We anticipate that during inference, by inputting the segmentation map of a synthetic image and prompting with 'a real picture of,' the model will convert the original image to the real domain. This approach is inspired by the Dreambooth fine-tuning method.

### Experiment 1: Direct Training ###

**Parameters:**

Batch size: 4 <br>
Learning rate: 1e-5 <br> 
Precision: 16 <br>
Accumulate gradient: 1 <br>
Image resolution: (512, 256), the size was reduced for acceleration reasons <br>
Epochs: 30 <br>
<br><br>
**Inference Results after training:**
<br>
![image](https://github.com/user-attachments/assets/2e77421a-3eb4-4a6e-b84e-d6c8fb86cf23)

In this initial training, it is evident that the model has learned to adapt the generated image to the input segmentation map and to transform synthetic images into the real domain, as indicated by the distinct style of the generated image compared to the synthetic theme. However, the quality remains subpar and is marred by numerous artifacts. 

### Experiment 2: Resolution scale-up ###
In this experiment, we aim to enhance the generative results by scaling the image resolution to (1024, 512). However, this adjustment has significantly slowed the training process, as we can no longer fit as many images into the GPU memory per batch. To circumvent the need to restart training at this slower rate, we utilized the weights from the last checkpoint—where the model had already learned control and context change—as a starting point for the new image resolution. This approach should reduce the number of training epochs required, as the model now only needs to adapt to the higher resolution.

Additionally, we implemented a random prompt deletion, removing the image descriptions for 20% of the images at random, to compel the model to infer the output solely from the segmentation map.

**Inference Results after training:**
![image](https://github.com/user-attachments/assets/c649e8c3-c28c-4102-a604-b35d7537f581)

In this second training session, it is evident that the generated images possess more detail and improved quality. However, some inferior examples were also produced. This discrepancy was somewhat troubling, as the training logs indicated a higher quality of generation. Upon investigating the potential causes for the decline in quality from training to inference, it was discovered that the segmentation maps for synthetic images had some noise, including segments not assigned to any class. This contrasts with real images that have precise annotations.

### Experiment 3: Control data augmentation ###

To solve the problem in the section above,we added some regularization techniques to prevent the model from overfitting. 
-we add random salt & peper noise in the image segments: classify a pixel in the class 0 (black) with a probability p to train the model on imperfect segmentations.<br>
  ![image](https://github.com/user-attachments/assets/c59b7531-1f65-4f15-a0b1-6657012da94e)

-we delete the label of some segments randomely to force the model to learn the object class just from its shape & the context and not from the color (label Id) <br>
  ![image](https://github.com/user-attachments/assets/3eb438d7-dfa2-47ed-8245-52602474e9c0)

In this experiment,we maintained the same parameters and the strategy of initially training on low-resolution images until the model achieved control, then progressing to higher resolutions. The sole change was adjusting the learning rate to 4e-5.

**Inference Results after training:**

- First stage of training: Results on low resolution: 
  ![image](https://github.com/user-attachments/assets/90f3d26b-ef83-4f71-bae8-2003d2aa3f89)

Generated images are already better even in the low resolution training, the regularization methods + higher learning rate did indeed help the model generalize better on synthetic data. 
  
- Second stage of training: **Results on higher resolution**

  ![image](https://github.com/user-attachments/assets/f3e7c1b6-3370-4548-8a25-91968810415d)

As anticipated from the initial results following low-resolution training, the image quality significantly improved upon scaling to higher resolutions. Further examination of additional samples of the generated images revealed that the regularization methods enhanced the model's ability to produce more accurate shapes of cars and vehicles overall. This is in contrast to previous experiments where the model had difficulties with details such as forming wheels. Up to this point, we have relied solely on visual inspection to assess image quality. However, for future comparisons, it would be prudent to explore other methods like [CLIP Image Quality Assessment (CLIP-IQA)](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html) or benchmarking techniques utilizing our [baseline classifier](https://github.com/bds-ailab/syn2real/blob/chore/opensourcing-project/12372-update_readme/docs/general_approach.md).

### Experiment 4: Adding Canny edges to control image ###

While image generation has significantly improved from our initial attempts, the model still produces numerous artifacts, particularly in expansive segments of buildings where it lacks sufficient control information. To address this, we considered implementing a secondary control using Canny images. Instead of integrating an additional ControlNet model for the Canny input, we opted to merge the segmentation maps with the Canny edges, as illustrated in the figure below. This approach reduces the number of trainable parameters and helps prevent overfitting given our limited dataset.

Canny Edges            |  Segmentation Map |  Canny Edges + Segmentation Map
:-------------------------:|:-------------------------: | :-------------------------:
![image](https://github.com/user-attachments/assets/1982bcc2-4b7f-4b3e-aec1-c2853128714a) | ![image](https://github.com/user-attachments/assets/deb7ccdc-ae30-4988-9ad1-9920d6a77ff6) | ![image](https://github.com/user-attachments/assets/ae5a68a5-6258-4e05-82d4-bba6ee3ab589)

In addition to incorporating Canny edge control, we trained the model using examples controlled solely by Canny edges or segmentation maps. This approach enables the model to assimilate information from both and adjust to imperfect control images during validation. Furthermore, we unfroze the decoder layers of Stable Diffusion to enhance generation quality. Although this action poses a risk of overfitting, the augmentation techniques we implemented prevented this outcome.

**Inference Results after training:** 

**NB:** the following results are a comparison on low resolution images (512, 256), we expect better results on higher resolutions.

Without Canny (Experiment 3 model)  |  With Canny + Unfrozen SD layers 
:-------------------------:|:-------------------------: 
![image](https://github.com/user-attachments/assets/12e429dd-7114-4db3-9f12-e126c05ac2cf) | ![image](https://github.com/user-attachments/assets/84019b33-e705-4276-b92f-268aaeefa23a)
![image](https://github.com/user-attachments/assets/038cffd5-61cf-4ea1-8da5-bddaeabc4bc5) | ![image](https://github.com/user-attachments/assets/eb9270a2-939f-4b7e-bb71-f2a5a22720c2)
![image](https://github.com/user-attachments/assets/57bf458f-7aa1-4aa8-81f2-d365e65ed52f) | ![image](https://github.com/user-attachments/assets/cc56a323-de7c-4270-bad5-f4ed95eaeb13)
![image](https://github.com/user-attachments/assets/fc838a4a-2be7-4cf3-aa78-be1326dfa883) | ![image](https://github.com/user-attachments/assets/3363db73-106d-4246-945d-d7076976202b)

### Experiment 5: Replacing the base model from SDv1.5 to SDv2.1 ###

SD v1.5         |  SD v2.1 
:-------------------------:|:-------------------------: 
![image](https://github.com/user-attachments/assets/6b555c3d-88f8-4e0d-9d37-c4716394a681) | ![image](https://github.com/user-attachments/assets/78f34fa5-5dce-4e66-b449-d1d5680d50a1)
![image](https://github.com/user-attachments/assets/b7fcb738-aa8f-4000-9123-73f12de98dc4) | ![image](https://github.com/user-attachments/assets/17747870-3346-4be3-9f98-9fdaa27d48bf)
![image](https://github.com/user-attachments/assets/6d7dd542-1826-4d28-8308-f9b9f4c5ae8e) | ![image](https://github.com/user-attachments/assets/ea2b0bb4-5e0f-457b-8056-fcc7085fb052)

### Experiment 6: Upscaling images to super quality using DeepFloyd IF ###

Synthetic Image          |  ControlNet generated image  |  IF Upscaled Image
:-------------------------:|:-------------------------: | :-------------------------:
![image](https://github.com/user-attachments/assets/eb18e733-7ae7-415e-870a-c8e4a773ee90) | ![image](https://github.com/user-attachments/assets/2e8f3465-2b8e-4bae-9d0d-3762c3614f49) | ![image](https://github.com/user-attachments/assets/e7e562d1-e85e-4446-a5a0-feeedb1d002a)
![image](https://github.com/user-attachments/assets/d62337db-6b82-4353-b3e4-3cf1db35022e) | ![image](https://github.com/user-attachments/assets/1abdd3c0-9ec7-40a3-b864-9485d7e97513) | ![image](https://github.com/user-attachments/assets/93b10a37-915f-488d-a950-5ef2e500a405)

### Experiment 7: Using SDXL as base model instead of SDv2.1/v1.5 ###

**First results before unlocking SD decoder layers and without DeepFloyd upscaling**

![image](https://github.com/user-attachments/assets/5f4a7039-b9d4-492d-bbe8-4c0cf96eb3e0)


### Experiment 8: Training SDXL with Active Learning on new transformed images ###

**Generation Variability Test: (3 training rounds)**

 Synthetic Image         |  Segmentation Map
:-------------------------:|:-------------------------: 
![image](https://github.com/user-attachments/assets/5029afd5-0847-4dc9-bcff-6c0962377259) | ![image](https://github.com/user-attachments/assets/61177cb8-c9f3-4c34-9ba1-00afe5136ae8)

**Generated examples:**

![image](https://github.com/user-attachments/assets/8a2575f2-3d30-4323-9e42-5ca503728d0e)

**More results:**

![image](https://github.com/user-attachments/assets/ad901554-1955-4296-b37f-82d500d429eb)


#### Experiment X: Comparison with CycleGAN-Turbo ####

![image](https://github.com/user-attachments/assets/fffc3bb0-781d-433f-bdad-a0628e8dacd2)












