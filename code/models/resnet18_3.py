"""
Cloned from https://github.com/julianstastny/VAE-ResNet18-PyTorch
Adjusted to make it more modular for several datasets.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1, layer = 1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # if layer ==4 and stride ==1:
        #     #last convolutional layer
        #     self.conv2 = nn.Sequential()
        #     self.bn2 = nn.Sequential()
        # else:
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec_transposed(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()
        self.stride = stride
        planes = int(in_planes/stride)

        self.conv1 = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)




        if stride == 1:
            self.conv2 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv2 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, padding=1 , bias=False, output_padding=1)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, output_padding=1),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # print("Michael jackson", self.stride)
        out = torch.relu(self.bn1(self.conv1(x)))
        # print("after first conv", out.size())
        out = self.bn2(self.conv2(out))
        out_temp = self.shortcut(x)
        # print("out, shortcut", out.size(), out_temp.size())
        out += out_temp
        out = torch.relu(out)
        return out



class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, x_dim=28, nc=1):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2, layer = 4)

        # self.linear1 = nn.Linear(32768, 1024)
        # self.linear2 = nn.Linear(1024, 2 * z_dim)
        # self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 2 * z_dim)
    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride, layer = 1):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, layer)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        # print("enc: siz x1 is",x.size())
        x = self.bn1(x)
        x = torch.relu(x)
        # print("enc: siz x2 is",x.size())
        x = self.layer1(x)
        # print("enc: siz x3 is",x.size())
        x = self.layer2(x)
        # print("enc: siz x4 is",x.size())
        x = self.layer3(x)
        # print("enc: siz x5 is",x.size())
        x = self.layer4(x)
        # print("enc: siz x6 is",x.size())
        x = F.adaptive_avg_pool2d(x, 1)
        # print("enc: siz x7 is",x.size())
        x = x.view(x.size(0), -1)
        # print("enc: siz x8 is",x.size())
        # x = self.linear1(x)
        # print("enc: siz x9a is",x.size())
        x = self.linear2(x)
        # print("enc: siz x9b is",x.size())

        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        # print("mu, logvar",mu.size(), logvar.size())
        return mu, logvar


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


class vanilla_decoder(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, x_dim=28, nc=1):
        super(vanilla_decoder, self).__init__()
        self.nc = nc
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.net1 = nn.Sequential(
                nn.Linear(z_dim, 1024),
                nn.ReLU(),

                nn.Linear(1024, 16384), # 1024, 4, 4
                Unflatten(1024, 4, 4),
                nn.ReLU(),

                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )

    def forward(self, z):
        x = z
        for layer in self.net1:
            x = layer(x)

        x = x.view(x.size(0), self.nc , self.x_dim, self.x_dim)
        return x
class ResNet18VAE_3(nn.Module):

    def __init__(self, z_dim, x_dim =28, nc = 3):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim, x_dim=x_dim, nc=nc)
        self.decoder = vanilla_decoder(z_dim=z_dim, x_dim=x_dim, nc=nc)

        self.mean = [0.4305, 0.3999, 0.3900]
        self.std = [0.1822, 0.1733, 0.1624]
        self.mv_normalize = T.Compose([T.Normalize(mean = self.mean, std = self.std)])

    def forward(self, x):
        x = self.mvtec_normalize(x)
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        dec_val = self.decoder(z)
        return dec_val , mean, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z


    def mvtec_normalize(self, x):
        norm = self.mv_normalize(x)
        return norm

    @staticmethod
    def reparameterize_eval(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
