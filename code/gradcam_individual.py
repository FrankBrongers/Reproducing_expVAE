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
        
    def generate(self):
        """
        Generates attention map and all individual maps per latent dimension.
        """

        self.A = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)

        b, n, w, h = self.A.shape

        M_list = torch.zeros([self.z.shape[1], b, self.image_size, self.image_size]).cuda()
        print('Mlist shape', M_list.shape )
        print('z shape', self.z.shape)
        
        for i, z_i in enumerate(self.z[1]):
            self.grads = self.get_conv_outputs(self.outputs_backward, self.target_layer)

            one_hot = torch.zeros_like(self.z)

            one_hot[:,i] = 1

            self.z.backward(gradient = one_hot, retain_graph=True)

            gradients = self.grads.cpu().data.numpy()[0]
            a_k = np.sum(gradients, axis=(1,2)) / (gradients.shape[1] * gradients.shape[2])

            M_i = torch.zeros_like(self.A[:, 1, :, :])

            for k in range(n):
                M_i += F.relu(a_k[k] * self.A[:, k, :, :])
            
            M_i = M_i.view(M_i.shape[0], 1, M_i.shape[1], M_i.shape[2])
            M_i = F.interpolate(M_i, (self.image_size, self.image_size),
                                    mode="bilinear", align_corners=True)
            M_i = M_i.squeeze(1)
            M_list[i, :, :, :] += M_i
            print(M_list.shape)

        M = torch.mean(M_list, dim=0)
        M = M.squeeze(1)

        return M, M_list
