# Weather4cast2021-SwinEncoderDecoder (AI4EX Team)

## Table of Content
* [General Info](#general-info)
* [Requirements](#requirements)
* [Usage](#usage)

## General Info
The resipository contains the code and learned model parameters for our submision in Weather4cast2021 stage-1 competition.

## Requirements
This resipository depends on the following packages availability
- Pytorch Lightning
- timm
- torch_optimizer
- pytorch_model_summary
- einops

## Usage
- a.1) train from scratch
    ```
    python main.py --gpus 0 --use_all_region
    ```
- a.2) fine tune a model from a checkpoint
    ```
    python main.py --gpu_id 1 --use_all_region --mode train --name ALL_real_swinencoder3d_688080 --time-code 20210630T224355 --initial-epoch 58```
    
- b.1) evaluate an untrained model (with random weights)
    ```
    python main.py --gpus 0 --use_all_region --mode test
    ```
- b.2) evaluate a trained model from a checkpoint (submitted inference)
    ```
    python main.py --gpu_id 1 --use_all_region --mode test --name ALL_real_swinencoder3d_688080 --time-code 20210630T224355 --initial-epoch 58
    ```
    
