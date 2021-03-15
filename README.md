# Reproductionality Challenge 2020-2021 - Towards Visually Explaining Variational Autoencoders

## Overview
This repository provides training and testing code and data for [paper](https://arxiv.org/pdf/1911.07389.pdf):

"Towards Visually Explaining Variational Autoencoders", Wenqian Liu, Runze Li, Meng Zheng, Srikrishna Karanam, Ziyan Wu, Bir Bhanu, Richard J. Radke, and Octavia Camps


## Requirements
```
python 3.8.5
pytorch 1.7.0
torchvision 0.8.1
opencv 4.5.0
matplotlib 3.3.3
tqdm 4.56.0
```
You can easily install al dependencies with anaconda using <br>
```
conda env create -f environment.yml
```

## Running codes
Please use the corresponding notebook file to run all desired experiments. See ```Anomaly_Detection/``` for the implementation of the anomaly detection and ```Latent_Space_Disentanglement/``` for the implementation of the latent space disentanglement, and a more detailed description.
