# CWPrompt-main

## Introduction
This is our implementation of our paper *Class-Weighted Prompt Learning for Rehearsal-Free Class-Incremental Learning*.

**TL;DR**: Class-Weighted Prompt Learning for Rehearsal-Free Class-Incremental Learning

**Abstract**:
Class-incremental learning (CIL) is a machine learning paradigm, which requires models continually adapt to new data and classes, with unknown task boundary during inference. With the development of large models, prompt learning methods which incorporate small size of prompts into the pre-trained models for instructing the model to adapt sequential tasks, become popular to CIL. Existing prompt learning methods suffer from an unavoidable plasticity-stability issue: the task-level nature of prompts prevent models from capturing inter-class knowledge during training, limiting the plasticity; while the prompt-key matching strategy of prompts risks on the incorrect prompt selection. We argue that the essence of the plasticity-stability issue is the flawed design of learnable prompts, which cannot balance the high discrimination of training prompts and the mismatch of prompt selection during inference. In this paper, we propose a class-weighted prompt framework (CWPrompt) for alleviating the plasticity-stability issue of rehearsal-free CIL. The core ingredients of CWPrompt are the task text prompt (TTP) module and the class-weight prompt (CWP) module. The former encodes the task text descriptions with task-level text prompts using a pre-trained text encoder, which promotes the correctness of prompt selection by unifying diverse class-specific tasks within a shared linguistic space. The latter utilizes class-level text prompts and class-level visual prompts for generating an instance-level multi-adaptive-knowledge prompt, which accumulates all the class-prominent knowledge together by adaptively allocating different class-dependent weights to each class-level visual prompts. Thanks to the weighted summation strategy, the generated multi-adaptive-knowledge prompt not only covers diverse multiple knowledge of all the classes in each task, but also flexibly adapts to each instance with its class-prominent weight. Experimental results on four datasets demonstrate the effectiveness of CWPrompt in both the CIL performance and prompt matching accuracy.


## Dependencies
- matplotlib==3.5.3
- numpy==1.20.3
- Pillow==10.3.0
- requests==2.26.0
- scikit_learn==1.4.1.post1
- scipy==1.7.1
- sentence_transformers==2.6.1
- six==1.16.0
- submitit==1.5.1
- timm==0.6.7
- torch==2.0.1
- torchvision==0.15.2
- transformers==4.39.3



## Usage

##### 1. Install dependencies
First we recommend to create a conda environment by using the following command.
```
conda create -n CWPrompt 
```
This command creates a conda environment named `CWPrompt`. You can activate the conda environment with the following command:
```
conda activate CWPrompt
```
The code is written for python `3.8.18`, but should work for other version with some modifications.
```
pip install -r requirements.txt



##### 2. Run code
- CIFAR-100
    ```
    python -m main cifar100_cwprompt \
        --num_tasks 10 \
        --data-path ./data/ \
        --output_dir ./output 
    ```

- ImageNet-R
    ```
    python -m main imr_cwprompt \
        --num_tasks 10 \
        --data-path ./data/ \
        --output_dir ./output
    ```

- CUB200
    ```
    python -m main cub_cwprompt \
        --num_tasks 10 \
        --data-path ./data/ \
        --output_dir ./output
    ```

- EuroSAT
    ```
    python -m main eurosat_cwprompt \
        --num_tasks 5 \
        --data-path ./data/ \
        --output_dir ./output
    ```

## Parameters

| Parameter         |           Description                       | 
|-------------------|---------------------------------------------|
| dataset              |   Dataset to use                            |
| num_tasks           |   Number of tasks |
| device              |   GPU device ID (default: 0)                |
| batch_size        |   batch size for training    |
| epochs            |   epochs                    |

## 
