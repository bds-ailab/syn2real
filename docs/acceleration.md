## Acceleration

### Training Pipeline
Typically, the training of deep learning models consists of two main parts:

- Data Loading: This occurs on the CPU, where datasets are shuffled and batched randomly, then loaded onto the computing devices using DataLoaders.
- Inference & Training: This is carried out on the GPU, where forward and backward passes are executed to compute the updated parameters.

During the development phase, computation time and performance are often overlooked. However, when transitioning to production and scaling, acceleration techniques become crucial to make training more feasible, particularly for large models. In this tutorial, I will introduce some straightforward techniques that can be applied in PyTorch (as an example) to boost training performance with minimal code changes (all techniques are fully supported by the PyTorch library).
![image](https://github.com/user-attachments/assets/9a4b88ec-20db-4ee0-b544-336b7a66389e)

### Data loading acceleration
#### Workers Number
Generally, data loading constitutes a bottleneck in the training pipeline; the larger the batch size and image size, the more challenging it is for a single CPU thread to load data onto the computing device (GPU). As illustrated in the figure below, the data loading time significantly surpasses the computation time for a standard benchmarking model, ResNet50, operating on a single GPU with a batch size of 32.

![image](https://github.com/user-attachments/assets/41600d4c-1287-45ae-b9aa-15fb885e93aa)

To address the bottleneck issue, we employ multiple workers who alternate loading data into the GPU. For example, with two workers, one loads data from disk storage while the other prepares the next data batch. Once the GPU finishes computing, it receives the ready batch from the first worker, and by the time it's done, the second worker is ready to transfer the next batch. This simplified explanation illustrates that while two workers may not eliminate the blocking time entirely, they can significantly reduce it. Below is a basic method to set the number of workers in PyTorch DataLoaders:

```
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
```

![image](https://github.com/user-attachments/assets/58a8ba48-5b9e-4387-a842-d797890d49e3)


By adjusting the number of workers to the appropriate proportion based on your training parameters, you can completely eliminate bottleneck time; however, there are several considerations to keep in mind:

- **Shared memory (shm):** To manage the data transfer between multiple loader workers and the main process, Pytorch employs the multiprocessing module. This module requires ample shared memory to operate effectively and to enable direct access to the memory space where data is stored, thus avoiding the need for serialization, which can decelerate the process. Therefore, the shared memory size must be adequate for the multiprocessing of the designated number of workers. In Docker containers, the default shared memory size is 64MB if not otherwise specified, which may be insufficient for multiprocessing since Docker threads also consume a portion of this memory. The shared memory size for Docker containers can be configured in the docker-compose.yaml file using specific parameters.
                               
```
version: x.x
services:
  your_service:
    build:
      context: .
      shm_size: '2gb' <-- this will set the size when building the container
    shm_size: '2gb' <-- when running the container
```

- **Disk IO bandwidth:** The bandwidth of storage disk IO is a physical limitation that prevents the use of an excessively large number of workers, as each thread requires data to be loaded from the disk. Typically, it is unnecessary to approach this limit.

### Pinned Memory & Non Blocking transfer

To transfer data to the GPU, it is initially moved from pageable memory to pinned memory on the CPU. Simply put, a worker retrieves the batch from storage disk after shuffling, applies the required preprocessing transformations, and waits for the GPU to finish its computations and be ready for the next batch. Once the GPU is ready, the batch is transferred from the CPU's memory to pinned memory and then to the GPU. This process requires the worker to wait for the GPU to complete its tasks. To streamline this, enabling pin_memory in PyTorch DataLoaders allows data to be directly stored in pinned memory after preprocessing. With the worker now available to retrieve the next batch, activating the non_blocking feature permits the worker to proceed to fetch the next batch while the first one is loaded into pinned memory.

![image](https://github.com/user-attachments/assets/adbae685-75dc-44c6-a718-c98020691f2e)
![image](https://github.com/user-attachments/assets/7ae38edd-371f-4f63-a173-f85a6241b975)


Here's how these parameters can be easily added to your code: 


```
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pinned_memory=True)

...
...

for inputs, labels in dataloader:
		    model.train()
            # Attaching batch data to device
            # Note: non blocking is used to optimize the waiting time for batch transfer to GPU
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

...
```

**Remark:** In the preceding section, I only outlined some techniques applicable to nearly all data loading processes. Nonetheless, additional technical decisions can optimize this phase, such as selecting a data storage format. Some formats may decode faster but require more storage space. Such decisions should be based on individual circumstances, constraints, and priorities. 

### Gradient Computing & Mixed Precision

By default, a model's parameters, activations, and gradients are represented in Float32 format. To reduce memory usage and increase throughput, one might consider a half-precision (Float16) representation. However, this approach presents two challenges:

Firstly, representing the model's weights in FP16 format can lead to precision loss when computing the model's output.
Secondly, gradients with very small values cannot be represented in FP16.
To address these issues while still benefiting from FP16, we employ the following techniques:

#### Mixed Precision

Model weights are maintained in FP32 to preserve accuracy and prevent the precision loss that could occur if weights were stored in FP16. However, most computations during the forward and backward passes are performed in FP16. This significantly accelerates operations due to reduced computational and memory demands. Although some precision is lost in the gradient computations for updating the parameters, this minor loss does not impact overall training. Since gradients are inherently approximate (stochastic gradient descent), the optimizer will adjust for any directional errors in subsequent steps.

![image](https://github.com/user-attachments/assets/1b8de982-8b16-4ecb-9553-165f667c08ca)


#### Loss Scaling
To prevent underflow (values too small for representation in FP16) during backpropagation, loss scaling (or gradient scaling) is utilized. The loss is scaled up by a large factor before backpropagation, and subsequently, gradients are scaled down by the same factor post-backpropagation. 

Incorporating these two features is straightforward, akin to earlier additions, as PyTorch already includes a scaler object and a context manager function for mixed precision.

```python
from torch.cuda.amp import GradScaler, autocast

# Init Gradient scaler
scaler = GradScaler()

# Iterate on epochs number
for epoch in range(num_epochs):
	# Setting the model to training mode
    model.train()
    # Optimization for each batch
    for inputs, labels in train_dataloader:
        # Attaching batch data to device cuda or cpu
        # Note: non blocking is used to optimize the waiting time for batch transfe to GPU
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Using autocast as context manager to allow running in mixed precision
        with autocast():
             # Forward pass
             outputs = model(inputs)
             loss = criterion(outputs, labels)

        # Backward pass and optimize
        # Using scaler to rescale back the updated parameters
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```


#### Benchmarking example:

To demonstrate the acceleration gain using only these simple techniques (Multiple workers + Asynchronous data transfers + Mixed precision & Grad scaler), I used a ResNet50 model (famous for benchmarking), Trained in the following conditions: 

- Dataset size: 60k images 
- Batch size: 32 
- Number of epochs: 6
- Learning rate: 0.001
- Optimizer: Adam
- Workers: 4
- Preprocessing Operations: Resizing and data augmentation transforms. 
- Added one fully connected layer for labels classification. (12 classes → 12 neurones) 

**Results:** Training was accelerated *6 times going down from 1H to 10min for the whole training (6 epochs). Accuracy and other evaluation metrics showed that the model is equally performing with or without these techniques which proves that no accuracy was lost at the expence of computation time.

### Advanced Parallelization Techniques

In this section, more advanced methods for accelerating computations will be discussed, necessitating the use of multiple GPU devices and/or multiple nodes. Through multiprocessing, several instances of the same program run on distinct computing devices with separate memory spaces. Assigning different data or code to these instances can expedite computation, assuming that tasks can be divided and consistent results achieved. Typically, this requires communication between the processes.

#### Distributed Data Prallelism (DDP)


[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) implements data parallelism at the module level which can run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP instance per process. DDP uses collective communications in the torch.distributed package to synchronize gradients and buffers. More specifically, DDP registers an autograd hook for each parameter given by model.parameters() and the hook will fire when the corresponding gradient is computed in the backward pass. Then DDP uses that signal to trigger gradient synchronization across processes. The recommended way to use DDP is to spawn one process for each model replica, where a model replica can span multiple devices. DDP processes can be placed on the same machine or across machines, but GPU devices cannot be shared across processes. 

![image](https://github.com/user-attachments/assets/e8cd8709-216c-49f4-8590-e04ff2c6c517)

Implementation Tutorial: [Getting Started with Distributed Data Parallel — PyTorch Tutorials 2.3.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

#### Pipeline Parallelism

Pipeline parallelism was original introduced in the Gpipe paper and is an efficient technique to train large models on multiple GPUs.

Typically for large models which don’t fit on a single GPU, model parallelism is employed where certain parts of the model are placed on different GPUs. Although, if this is done naively for sequential models, the training process suffers from GPU under utilization since only one GPU is active at one time as shown in the figure below:

![image](https://github.com/user-attachments/assets/4a6f5da6-9ae8-4dd0-9150-f6000f6c0f24)

The figure represents a model with 4 layers placed on 4 different GPUs (vertical axis). The horizontal axis represents training this model through time demonstrating that only 1 GPU is utilized at a time. 

To alleviate this problem, pipeline parallelism splits the input minibatch into multiple microbatches and pipelines the execution of these microbatches across multiple GPUs. This is outlined in the figure below:

![image](https://github.com/user-attachments/assets/c67134bc-cd2a-4583-88a9-0e30c60e58ae)

The figure represents a model with 4 layers placed on 4 different GPUs (vertical axis). The horizontal axis represents training this model through time demonstrating that the GPUs are utilized much more efficiently. However, there still exists a bubble (as demonstrated in the figure) where certain GPUs are not utilized.<

Implementation Tutorial: [Training Transformer models using Pipeline Parallelism — PyTorch Tutorials 2.2.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html)


#### Tensor Parallelism


Tensor Parallel (TP) was originally proposed in the [Megatron-LM](https://arxiv.org/abs/1909.08053) paper, and it is an efficient model parallelism technique to train large scale Transformer models. [Sequence Parallel](https://arxiv.org/abs/2205.05198) (SP) is a variant of Tensor Parallel that shards on the sequence dimension for nn.LayerNorm or RMSNorm to further save activation memory during training. As the model becomes larger, the activation memory becomes the bottleneck, so in Tensor Parallel training it usually applies Sequence Parallel to LayerNorm or RMSNorm layers.

The PyTorch Fully Sharded Data Parallel (FSDP) already has the capability to scale model training to a specific number of GPUs. However, when it comes to further scale the model training in terms of model size and GPU quantity, many additional challenges arise that may require combining Tensor Parallel with FSDP.:

1. As the world size (number of GPUs) is becoming excessively large (exceeding 128/256 GPUs), the FSDP collectives (such as allgather) are being dominated by ring latency. By implementing TP/SP on top of FSDP, the FSDP world size could be reduced by 8 by applying FSDP to be inter-host only, consequently decreasing the latency costs by the same amount.

2. Hit data parallelism limit where you can not raise the global batch size to be above the number of GPUs due to both convergence and GPU memory limitations, Tensor/Sequence Parallel is the only known way to “ballpark” the global batch size and continue scaling with more GPUs. This means both model size and number of GPUs could continue to scale.

3. For certain types of models, when local batch size becomes smaller, TP/SP can yield matrix multiplication shapes that are more optimized for floating point operations (FLOPS).

So, when pre-training, how easy is it to hit those limits? As of now, pre-training a Large Language Model (LLM) with billions or trillions of tokens could take months, even when using thousands of GPUs.

- It will always hit limitation 1 when training LLM on a large scale. For example, Llama 2 70B trained with 2k GPUs for 35 days, multi-dimensional parallelisms are needed at 2k scale.

- When the Transformer model becomes larger (such as Llama2 70B), it will also quickly hit the limitation 2. One could not use FSDP alone with even local batch_size=1 due to memory and convergence constraints. For example, Llama 2 global batch size is 1K, so data parallelism alone can not be used at 2K GPUs.

Implementation Tutorial: [Large Scale Transformer model training with Tensor Parallel (TP) — PyTorch Tutorials 2.3.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)

**Remark**: Pytorch-Lightning offers even simpler implementations of these parallelization techniques.


### Fine Tuning Optimization: Low-Rank Adaptation (LoRA)


Now, focusing on a more specific training task: fine-tuning or transfer learning. To minimize the number of parameters updated in this step, various approaches add trainable layers at the end of the model, before, or between the self-attention multiheads. LoRA represents a different paradigm, seeking the least number of the model's parameters modifications to achieve comparable results.

To make fine-tuning more efficient, LoRA’s approach is to represent the weight updates with two smaller matrices (called update matrices) through low-rank decomposition. These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. The original weight matrix remains frozen and doesn’t receive any further adjustments. To produce the final results, both the original and the adapted weights are combined.

This approach has a number of advantages:

- LoRA makes fine-tuning more efficient by drastically reducing the number of trainable parameters.
- The original pre-trained weights are kept frozen, which means you can have multiple lightweight and portable LoRA models for various downstream tasks built on top of them.
- LoRA is orthogonal to many other parameter-efficient methods and can be combined with many of them.
- Performance of models fine-tuned using LoRA is comparable to the performance of fully fine-tuned models.
- LoRA does not add any inference latency because adapter weights can be merged with the base model.
In principle, LoRA can be applied to any subset of weight matrices in a neural network to reduce the number of trainable parameters. However, for simplicity and further parameter efficiency, in Transformer models LoRA is typically applied to attention blocks only. The resulting number of trainable parameters in a LoRA model depends on the size of the low-rank update matrices, which is determined mainly by the rank r and the shape of the original weight matrix.

![image](https://github.com/user-attachments/assets/6c772f7a-d115-4052-be5f-5a2fcf8150ef)

LoRA implementation is not very difficult even from scratch since you will only need to modify the forward method of your model by introducing the matrices A & B, however, there are already libraries that manage these modification such as transformers by HuggingFace. Implementation example: 

```
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-base-model", device_map = 'cuda')

config = LoraConfig(
                      r=32,
                       lora_alpha=32,
                       target_modules=["query", "value"],
                       lora_dropout=0.1,
                       bias="lora_only",
                       modules_to_save=["decode_head"],
               )
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)
```

### References
1- [Formation FIDLE (CNRS)](https://fidle.cnrs.fr/w3/)

2- [PyTorch documentation — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/index.html)

3- [Hugging Face - Documentation](https://huggingface.co/docs)
