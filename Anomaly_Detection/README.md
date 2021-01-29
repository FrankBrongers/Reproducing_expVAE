# expVAE - Anomaly detection
Pytorch implementation of VAE anomaly detection proposed in Towards Visually Explaining Variational Autoencoders (https://arxiv.org/abs/1911.07389).


## Train
To train the network the file train_expVAE.py can be run using the following line as an example:
```
python code/train_expVAE.py --dataset=mvtec_ad --model=resnet18_3 --batch_size=8 --one_class=5
```
The train and test functions are build to support the three VAE models: vanilla_mnist, vanilla_ped1 and resnet18_3.


# Test
To train the network the file train_expVAE.py can be run using the following line as an example:
```
python code/test_expVAE.py --dataset mvtec_ad --model resnet18_3 --batch_size 8 --model_path ./ckpt/resnet18_3_mvtecClass_5_checkpoint.pth --one_class 5 --target_layer encoder.layer2.1.conv1
```
