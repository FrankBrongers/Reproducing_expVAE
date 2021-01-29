from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
import os


class PropBase(object):
    def __init__(self, model, target_layer, device, image_size):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

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
            self.outputs_backward[id(module)] = grad_out

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

    def generate(self, z):
        """
        Generates attention map and all individual maps per latent dimension z.
        """

        A = self.get_conv_outputs(self.outputs_forward, self.target_layer)

        b, n, w, h = A.shape
        A_flat = A.view(b*n, w*h)

        M_list = torch.zeros([z.shape[1], b, self.image_size, self.image_size]).to(self.device)

        for i, z_i in enumerate(z[1]):
            one_hot = torch.zeros_like(z)
            one_hot[:,i] = 1
            self.model.zero_grad()
            z.backward(gradient=one_hot, retain_graph=True)

            self.grads = self.get_conv_outputs(self.outputs_backward, self.target_layer)

            gradients = self.grads[0].to(self.device)
            a_k = (torch.sum(gradients, dim=(2,3)) / (gradients.shape[2] * gradients.shape[3])).flatten()

            a_kA = torch.multiply(a_k, A_flat.T).T
            a_kA = (a_kA).view((b, n, w, h))
            M_i = F.relu(a_kA).sum(dim=1, keepdim=True)
            M_i = F.interpolate(M_i, (self.image_size, self.image_size),
                                    mode="bilinear", align_corners=True)
            M_i = M_i.squeeze(1)
            M_list[i, :, :, :] += M_i

        return M_list
