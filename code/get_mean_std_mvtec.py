#!/bin/bash
import argparse
import torch
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc

import os
import numpy as np
import matplotlib.pyplot as plt

from models.vanilla import ConvVAE
from models.vanilla_ped1 import ConvVAE_ped1
from models.resnet18 import ResNet18VAE
from models.resnet18_2 import ResNet18VAE_2

import OneClassMnist
import Ped1_loader
import MVTec_loader as mvtec

from gradcam import GradCAM
import cv2
from PIL import Image
from torchvision.utils import save_image, make_grid

    # for dataloader check: pin pin_memory, batch size 32 in original
mean = 0.
std = 0.
totlen =0
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

num_classes = len(CLASS_NAMES)
for i in range(1):
    class_name = mvtec.CLASS_NAMES[5]   # nuts
    train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True, grayscale=False)
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, **kwargs)



    for images, _ in train_loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    totlen += len(train_loader.dataset)
    print(i)
mean /= totlen
std /= totlen
print("mean, std",mean, std)
