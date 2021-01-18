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

    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def forward(self, x):
        self.preds = self.model(x)
        self.image_size = x.size(-1)
        # Note: model moet ook z als output returnen, anders werkt niet. 
        recon_batch, self.mu, self.logvar, self.z = self.model(x)
        return recon_batch, self.mu, self.logvar


    def backward(self, mu, logvar, mu_avg, logvar_avg):
        self.model.zero_grad()

        mu = mu.to(self.device)
        self.score_fc = torch.sum(mu)
        self.score_fc.backward(retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer):
        """
        Retrieves model output for a specific module from a dictionary.
        Inputs:
            outputs - Dictionary to retrieve values from (forward or backward)
            target_layer - Specific module for which to retrieve values
        """
        return list(outputs.values())[0]

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
            # module[0] is name [1] is the module itself
            if module[0] == self.target_layer :
                module[1].register_backward_hook(func_b)
                module[1].register_forward_hook(func_f)
        
    def generate_one(self):
        """
        Generates attention map for 1 image.
        """

        self.grads = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)

        self.A = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)

        # Kan nu de batches nog niet handlen.
        A = self.A.squeeze(0)
        z = self.z.squeeze(0)
        n, w, h = A.shape

        M_list = torch.zeros([z.shape[0], w, h]).cuda()
        
        for i, z_i in enumerate(z):
            
            one_hot = torch.zeros_like(z)
            one_hot[i] = 1
            z.backward(gradient = one_hot, retain_graph=True)

            gradients = self.grads.cpu().data.numpy()[0]

            a_k = np.sum(gradients, axis=(1,2)) / (gradients.shape[1] * gradients.shape[2])

            M_i = torch.zeros_like(A[1, :, :])
            for k in range(n):
                M_i += F.relu(a_k[k] * A[k, :, :])
            M_list[i, :, :] += M_i

        M = torch.mean(M_list, dim=0)

        #TODO: interpolate/upsample M

        return M, M_list
