"""main.py"""

import argparse
import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = Solver(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='(AD) Factor-VAE')

    parser.add_argument('--seed', default=0, type=int, help='the seed')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=300000, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=40, type=float, help='gamma hyperparameter')
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')

    parser.add_argument('--ad_loss', type=bool, const=True, default=False, nargs='?', help='add if the attention disentanglement loss should be used')
    parser.add_argument('--target_layer', type=str, default='0', help='target layer for the attention maps')
    parser.add_argument('--lamb', default=1, type=float, help='lambda hyperparameter for the attention disentanglement loss')

    parser.add_argument('--L', default=100, type=int, help='amount of samples used for creating the votes in the disentanglement score')
    parser.add_argument('--vote_count', default=800, type=int, help='amount of votes needed for the disentanglement score')
    parser.add_argument('--dis_score', type=bool, const=True, default=False, nargs='?', help='add if the disentanglement score should be measured')
    parser.add_argument('--dis_batch_size', default=2048, type=int, help='batch size for the disentanglement score')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='dsprites', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')
    parser.add_argument('--print_iter', default=500, type=int, help='print losses iter')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_save_iter', default=300000, type=int, help='checkpoint save iter')

    parser.add_argument('--results_dir', default='results', type=str, help='results directory, saves the arguments and outputs')
    parser.add_argument('--results_save', default=True, type=str2bool, help='whether to save the results')

    args = parser.parse_args()

    main(args)
