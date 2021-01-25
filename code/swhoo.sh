#!/bin/sh

for layer in encoder.layer1.0.conv1 encoder.layer1.0.conv2 encoder.layer1.1.conv1 encoder.layer1.1.conv2 encoder.layer2.0.conv1 encoder.layer2.0.conv2 encoder.layer2.1.conv1 encoder.layer2.1.conv2 encoder.layer3.0.conv1 encoder.layer3.0.conv2 encoder.layer3.1.conv1 encoder.layer3.1.conv2 encoder.layer4.0.conv1 encoder.layer4.0.conv2 encoder.layer4.1.conv1 encoder.layer4.1.conv2 encoder.layer2.0.shortcut.0 encoder.layer3.0.shortcut.0 encoder.layer4.0.shortcut.0
do
python3 test_expVAE.py --dataset mvtec_ad --model resnet18_2 --batch_size 2 --decoder resnet --model_path ./ckpt/resnet18_2_final_resDec.pth --target $layer
done
# encoder.layer2.0.shortcut.0 encoder.layer3.0.shortcut.0 encoder.layer4.0.shortcut.0
# python3 test_expVAE.py --dataset mvtec_ad --model resnet18_2 --batch_size 2 --model_path ./ckpt/resnet18_2_final_vanilDec.pth --target_layer encoder.layer3.0.conv2
# python3 test_expVAE.py --dataset mvtec_ad --model resnet18_2 --batch_size 2 --model_path ./ckpt/resnet18_2_checkpoint.pth --target encoder.layer4.1.conv1
