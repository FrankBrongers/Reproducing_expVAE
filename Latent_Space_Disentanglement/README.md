# AD-FactorVAE
Pytorch implementation of AD-FactorVAE proposed in Towards Visually Explaining Variational Autoencoders (https://arxiv.org/abs/1911.07389).<br>
FactorVAE implementation based on https://github.com/1Konny/FactorVAE.
<br>

### Dependencies
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


### 2D Shapes(dsprites) Dataset
```
sh scripts/prepare_data.sh dsprites
```
The data directory structure should be like below<br>
```
.
└── data
    └── dsprites-dataset
        └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

### Usage
The results can be reproduced by starting jupyter notebook and running ```results.ipynb```<br>
You can reproduce the models used as follows
```
sh scripts/reproduce_models.sh
```
or you can run your own experiments using the default settings or by setting those manually, run the following to see the possible options
```
python main.py --help
```

### Visualization
You can create visualizations of the attention maps, for example
```
python visualizer.py  --name FactorVAE --target_layer 0
```
And plot the results, for example
```
python plotter.py --names 'gamma40 lambda40_gamma40' --all_plots
```
Again, use the ```--help``` flag to check all possibilities

### References
1. Disentangling by Factorising, Kim et al. (http://arxiv.org/abs/1802.05983)
2. Towards Visually Explaining Variational Autoencoders, Liu et al. (https://arxiv.org/abs/1911.07389)
