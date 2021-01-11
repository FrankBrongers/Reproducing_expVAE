from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import os


class PropBase(object):

    def __init__(self, model, target_layer, cuda=True):
        self.model = model
        self.device = cuda
        self.model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()


    def set_hook_func(self):
        raise NotImplementedError

    # set the target class as one others as zero. use this vector for back prop
    # def encode_one_hot(self, idx):
    #     one_hot = torch.FloatTensor(1, self.n_class).zero_()
    #     one_hot[0][idx] = 1.0
    #     return one_hot

    # TODO: change this function and ask Lezi what she was thinking here
    # set the target class as one others as zero. use this vector for back prop added by Lezi

    def encode_one_hot_batch(self, z, mu, logvar, mu_avg, logvar_avg):
        one_hot_batch = torch.FloatTensor(z.size()).zero_()
        # They return mu here. Probably one_hot_batch is not implemented yet.
        return mu

    def forward(self, x):
        self.preds = self.model(x)
        self.image_size = x.size(-1)
        recon_batch, self.mu, self.logvar = self.model(x)
        return recon_batch, self.mu, self.logvar

    # back prop the one_hot signal
    def backward(self, mu, logvar, mu_avg, logvar_avg):
        self.model.zero_grad()
        z = self.model.reparameterize_eval(mu, logvar).to(self.device)
        one_hot = self.encode_one_hot_batch(z, mu, logvar, mu_avg, logvar_avg)

        one_hot = one_hot.to(self.device)

        # TODO:  remove flags
        flag = 2
        if flag == 1:
            print('Flag is 1 (somehow?)')
            self.score_fc = torch.sum(F.relu(one_hot * mu))
        else:
            self.score_fc = torch.sum(one_hot)
        self.score_fc.backward(retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer):
        """
        Retrieves model output for a specific module from a dictionary.
        Inputs:
            outputs - Dictionary to retrieve values from (forward or backward)
            target_layer - Specific module for which to retrieve values
        """
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))

class GradCAM(PropBase):
    # hook functions to compute gradients wrt intermediate results
    # so dz/dL and dz/dL not dW/dL as usual
    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            """
            Hook call function that stores the backward pass gradients for every
            network module in a dictionary.
            """
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            """
            Hook call function that stores the forward pass output for every
            network module in a dictionary.
            """
            self.outputs_forward[id(module)] = f_output

        # Loop over all layers in the network and store outputs of forward
        # and backward passes
        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)

    def normalize(self, grads):
        """
        Applies L2 normalization to the gradients
        """
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()


    def compute_gradient_weights(self):
        """
        Applies the GAP operation to the gradients to obtain weights alpha.
        """
        self.grads = self.normalize(self.grads.squeeze())
        # Get height and width of attention maps
        self.map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(self.map_size)(self.grads)


    def generate(self):
        """
        Generates attention map from gradients.
        """
        # Retrieve gradients of backward pass for target layer
        self.grads = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)
        # compute weigths based on the gradient
        self.compute_gradient_weights()

        # Retrieve output of forward pass for target layer and set as activation
        self.activiation = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)

        self.weights.volatile = False
        self.activiation = self.activiation[None, :, :, :, :]
        self.weights = self.weights[:, None, :, :, :]
        gcam = F.conv3d(self.activiation, (self.weights.to(self.device)), padding=0, groups=len(self.weights))
        gcam = gcam.squeeze(dim=0)
        gcam = F.upsample(gcam, (self.image_size, self.image_size), mode="bilinear")
        gcam = torch.abs(gcam)

        return gcam
