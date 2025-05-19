## ControlNet FineTuning ## 

After conducting a sanity check by training ControlNet on a symbolic dataset of circles, our model is now prepared for fine-tuning on actual synthetic/real images. In these initial experiments, we will only train the ControlNet component of our general strategy, excluding the discriminator block. The goal is to ensure that the ControlNet block can distinguish between the two domains and is capable of converting images from the synthetic domain to the real domain. As expected, we anticipate generations with artifacts, which typically occur when attempting to control the diffusion process. However, we plan to address this issue with a second discriminator block that will refine the generation through adversarial training.

For training, we will utilize a dataset that resembles synthetic images from the GTA game and real images from the Cityscapes dataset. This dataset's advantage lies in the segmentation labels for each image, which are consistent across both the synthetic and real domains, sharing the same class IDs and color codes. To train the model, we captioned each image using the BLIP model and then appended the tokens 'a real picture of'/'a synthetic picture of' to guide the model's output generation. We anticipate that during inference, by inputting the segmentation map of a synthetic image and prompting with 'a real picture of,' the model will convert the original image to the real domain. This approach is inspired by the Dreambooth fine-tuning method.

### Experiment 1: Direct Training ###

**Parameters: **

Batch size: 4 
Learning rate: 1e-5 
Precision: 16 
Accumulate gradient: 1
Image resolution: (512, 256), the size was reduced for acceleration reasons
Epochs: 30 
**Inference Results after training: **
![image](https://github.com/user-attachments/assets/2e77421a-3eb4-4a6e-b84e-d6c8fb86cf23)

In this initial training, it is evident that the model has learned to adapt the generated image to the input segmentation map and to transform synthetic images into the real domain, as indicated by the distinct style of the generated image compared to the synthetic theme. However, the quality remains subpar and is marred by numerous artifacts. 

### Experiment 2: Resolution scale-up ###
In this experiment, we aim to enhance the generative results by scaling the image resolution to (1024, 512). However, this adjustment has significantly slowed the training process, as we can no longer fit as many images into the GPU memory per batch. To circumvent the need to restart training at this slower rate, I utilized the weights from the last checkpoint—where the model had already learned control and context change—as a starting point for the new image resolution. This approach should reduce the number of training epochs required, as the model now only needs to adapt to the higher resolution.

Additionally, I implemented a random prompt deletion, removing the image descriptions for 20% of the images at random, to compel the model to infer the output solely from the segmentation map.

**Inference Results after training:**
![image](https://github.com/user-attachments/assets/c649e8c3-c28c-4102-a604-b35d7537f581)

In this second training session, it is evident that the generated images possess more detail and improved quality. However, some inferior examples were also produced. This discrepancy was somewhat troubling, as the training logs indicated a higher quality of generation. Upon investigating the potential causes for the decline in quality from training to inference, it was discovered that the segmentation maps for synthetic images had some noise, including segments not assigned to any class. This contrasts with real images that have precise annotations.

### Experiment 3: Control data augmentation ###

To solve the problem in the section above, I added some regularization techniques to prevent the model from overfitting. 
- I add random salt & peper noise in the image segments: classify a pixel in the class 0 (black) with a probability p to train the model on imperfect segmentations.
  ![image](https://github.com/user-attachments/assets/c59b7531-1f65-4f15-a0b1-6657012da94e)

- I delete the label of some segments randomely to force the model to learn the object class just from its shape & the context and not from the color (label Id)
  ![image](https://github.com/user-attachments/assets/3eb438d7-dfa2-47ed-8245-52602474e9c0)

In this experiment, I maintained the same parameters and the strategy of initially training on low-resolution images until the model achieved control, then progressing to higher resolutions. The sole change was adjusting the learning rate to 4e-5.

**Inference Results after training:**

- First stage of training: Results on low resolution: 
  ![image](https://github.com/user-attachments/assets/90f3d26b-ef83-4f71-bae8-2003d2aa3f89)
