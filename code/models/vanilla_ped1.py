# from code.test_expVAE import main
import torch
import torch.nn as nn
import math

from functools import reduce
from operator import mul

from torch.nn.modules.batchnorm import BatchNorm2d


def get_linear_size(input_size, num_layers):
    """
    Returns the resulting linear layer size 
    Inputs:
        input_size - Input image size (int)
        num_layers - The number of conv layers containg strides (int)
    """
    for i in range(num_layers):
        input_size = math.floor(input_size / 2)
    return input_size


class Flatten(nn.Module):
    def forward(self, input):
        """
        Custom module that flattens the channel, height and width of an image.
        Inputs:
            input - Unflattened image of size [batch_size, channel, height, width]
        """
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):

    def __init__(self, channel, height, width):
        """
        Custom module that unflattens an image, restoring its channel, height and width.
        """
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        """
        Forward pass of unflatten module
        Inputs:
            input - Flattened image of size [batch_size, (channel x height x width)]
        """
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE_ped1(nn.Module):

    def __init__(self, latent_size, input_size, layer_config, batch_norm=True):
        """
        Encoder and Decoder network
        Inputs:
            latent_size - Dimensionality of the latent vector (int)
            input_size - Input image size (int)
            layer_config - List of the conv layer configuration [int*, int, int, int]
                            * is the number of colourbands
            batch_norm - Whether to use batch normalization or not (Bool)
        """
        super(ConvVAE_ped1, self).__init__()

        self.latent_size = latent_size
        self.input_size = input_size
        self.config = layer_config
        self.batch_norm = batch_norm
            

        self.encoder = nn.Sequential()
        # Layer 1
        self.encoder.add_module("Conv1", nn.Conv2d(self.config[0], self.config[1], kernel_size=4, stride=2, padding=1))
        if self.batch_norm:
            self.encoder.add_module("Bn1", nn.BatchNorm2d(self.config[0]))
        self.encoder.add_module("ReLU1", nn.ReLU())

        # Layer 2
        self.encoder.add_module("Conv2", nn.Conv2d(self.config[1], self.config[2], kernel_size=4, stride=2, padding=1))
        if self.batch_norm:
            self.encoder.add_module("Bn2", nn.BatchNorm2d(self.config[2]))
        self.encoder.add_module("ReLU2", nn.ReLU())

        # Layer 3
        self.encoder.add_module("Conv3", nn.Conv2d(self.config[2], self.config[3], kernel_size=4, stride=2, padding=1))
        if self.batch_norm:
            self.encoder.add_module("Bn3", nn.BatchNorm2d(self.config[3]))
        self.encoder.add_module("ReLU3", nn.ReLU())

        # From conv to latent space
        self.encoder.add_module("Flatten", Flatten())
        self.encoder.add_module("Linear", nn.Linear(36864, 1024))
        self.encoder.add_module("ReLU", nn.ReLU())

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)


        self.decoder = nn.Sequential()
        # From latent space to conv
        self.decoder.add_module("Linear1", nn.Linear(self.latent_size, 1024))
        self.decoder.add_module("ReLU1", nn.ReLU())
        self.decoder.add_module("Linear2", nn.Linear(1024, 36864))
        self.decoder.add_module("ReLU1", nn.ReLU())
        self.decoder.add_module("Unflatten", Unflatten(self.config[3], 12, 12))

        # Layer 3
        self.decoder.add_module("ReLU3", nn.ReLU())
        if self.batch_norm:
            self.encoder.add_module("Bn3", nn.BatchNorm2d(self.config[3]))
        self.decoder.add_module("Conv3", nn.ConvTranspose2d(self.config[3], self.config[2], kernel_size=5, stride=2, padding=1))

        # Layer 2
        self.decoder.add_module("ReLU2", nn.ReLU())
        if self.batch_norm:
            self.encoder.add_module("Bn2", nn.BatchNorm2d(self.config[2]))
        self.decoder.add_module("Conv3", nn.ConvTranspose2d(self.config[2], self.config[1], kernel_size=4, stride=2, padding=1))
        
        # Layer 1
        self.decoder.add_module("ReLU1", nn.ReLU())
        if self.batch_norm:
            self.encoder.add_module("Bn2", nn.BatchNorm2d(self.config[3]))
        self.decoder.add_module("Conv3", nn.ConvTranspose2d(self.config[1], self.config[0], kernel_size=4, stride=2, padding=1))

        self.decoder.add_module("Sigmoid", nn.Sigmoid())

    def encode(self, x):
        """
        Encoder forward pass
        Inputs:
            x - Input batch of images, with shape [batch_size, channel, height, width]
        Outputs:
            mu - Tensor of shape [batch_size, latent_size] representing the predicted mean of the latent distributions.
            log_var - Tensor of shape [batch_size, latent_size] representing the predicted log standard deviation
                      of the latent distributions.
        """
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        """
        Decoder forward pass
        Inputs:
            z - Latent vector of size [batch_size, latent_size]
        Outputs:
            x_hat - Prediction of the reconstructed image based on z, of shape [batch_size, channel, height, width]
        """
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu, logvar):
        """
        Perform the reparameterization trick to obtain the latent vector z from the inferred posterior.
        Inputs:
            mu - Mean of the inferred distribution
            log_var - Standard deviation of the inferred distribution
        Outputs:
            z - A sample of the distributions, with gradient support for both mean and std.
                The tensor should have the same shape as the mean and std input tensors.
        """        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            return z
        # What is the function of the else statement? Why would we need to decode mu and not z for the test data?
        else:
            return mu

    def reparameterize_eval(self, mu, logvar):
        #TODO: what is the function of this function??
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        ConvVAE forward pass. Applies the encoder, reparameterization trick and decoder.
        Inputs:
            x - Input batch of images, with shape [batch_size, channel, height, width]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
