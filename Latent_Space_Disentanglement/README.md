# AD-FactorVAE
Pytorch implementation of AD-FactorVAE proposed in Towards Visually Explaining Variational Autoencoders (https://arxiv.org/abs/1911.07389).<br>
FactorVAE implementation based on https://github.com/1Konny/FactorVAE.
<br>

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
