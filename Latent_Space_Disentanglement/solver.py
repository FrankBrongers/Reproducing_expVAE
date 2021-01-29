"""solver.py"""

import os
import numpy.random as random
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import mkdirs, save_args_outputs
from ops import recon_loss, ad_loss, kl_divergence, permute_dims
from model import FactorVAE, Discriminator
from dataset import return_data
from disentanglement import disentanglement_score
from gradcam import GradCAM


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.pbar = tqdm(total=self.max_iter)

        # Data
        assert args.dataset == 'dsprites', 'Only dSprites is implemented'
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader, self.dataset = return_data(args)

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        # Disentanglement score
        self.L = args.L
        self.vote_count = args.vote_count
        self.dis_score = args.dis_score
        self.dis_batch_size = args.dis_batch_size

        # Models and optimizers
        self.VAE = FactorVAE(self.z_dim).to(self.device)
        self.nc = 1

        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.nets = [self.VAE, self.D]

        # Attention Disentanglement loss
        self.ad_loss = args.ad_loss
        self.lamb = args.lamb
        if self.ad_loss:
            self.gcam = GradCAM(self.VAE.encode, args.target_layer, self.device, args.image_size)
            self.pick2 = True

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name+'_'+str(args.seed))
        self.ckpt_save_iter = args.ckpt_save_iter
        if self.max_iter >= args.ckpt_save_iter:
            mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)

        # Results
        self.results_dir = os.path.join(args.results_dir, args.name+'_'+str(args.seed))
        self.results_save = args.results_save

        self.outputs = {'vae_recon_loss': [], 'vae_kld': [], 'vae_tc_loss': [], 'D_tc_loss': [], 'ad_loss': [], 'dis_score': [], 'iteration': []}

    def train(self):
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_ad_loss = self.get_ad_loss(z)
                vae_kld = kl_divergence(mu, logvar)

                D_z = self.D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss + self.lamb*vae_ad_loss

                x_true2 = x_true2.to(self.device)
                z_prime = self.VAE(x_true2, no_dec=True)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)

                self.optim_D.zero_grad()
                D_tc_loss.backward()

                self.optim_VAE.step()
                self.optim_D.step()

                if self.global_iter%self.print_iter == 0:
                    if self.dis_score:
                        dis_score = disentanglement_score(self.VAE.eval(), self.device, self.dataset, self.z_dim, self.L, self.vote_count, self.dis_batch_size)
                        self.VAE.train()
                    else:
                        dis_score = torch.tensor(0)

                    self.pbar.write('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} ad_loss:{:.3f} D_tc_loss:{:.3f} dis_score:{:.3f}'.format(
                        self.global_iter, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), vae_ad_loss.item(), D_tc_loss.item(), dis_score.item()))

                    if self.results_save:
                        self.outputs['vae_recon_loss'].append(vae_recon_loss.item())
                        self.outputs['vae_kld'].append(vae_kld.item())
                        self.outputs['vae_tc_loss'].append(vae_tc_loss.item())
                        self.outputs['D_tc_loss'].append(D_tc_loss.item())
                        self.outputs['ad_loss'].append(vae_ad_loss.item())
                        self.outputs['dis_score'].append(dis_score.item())
                        self.outputs['iteration'].append(self.global_iter)

                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

        if self.results_save:
            save_args_outputs(self.results_dir, self.args, self.outputs)

    def get_ad_loss(self, z):
        if not self.ad_loss:
            return torch.tensor(0)

        z_picked = z[:, random.randint(0, self.z_dim, size=2)]
        M = self.gcam.generate(z_picked)

        return ad_loss(M.flatten(1), self.batch_size, self.pick2)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))
