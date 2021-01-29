import torch
import torch.nn as nn

from functools import reduce
from operator import mul


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


class ConvVAE_mnist(nn.Module):

    def __init__(self, latent_size):
        """
        Encoder and Decoder network
        Inputs:
            latent_size - Dimensionality of the latent vector
        """
        super(ConvVAE_mnist, self).__init__()

        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(6272, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6272),
            nn.ReLU(),
            Unflatten(128, 7, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

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
