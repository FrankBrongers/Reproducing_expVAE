import os, argparse
import numpy as np
import torch
import cv2
from torchvision.utils import save_image, make_grid
from model import FactorVAE

from dataset import return_data
from gradcam import GradCAM


def load_checkpoint(model, ckpt_dir, ckptname, device, verbose=True):
    filepath = os.path.join(ckpt_dir, ckptname)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)

        model.load_state_dict(checkpoint['model_states']['VAE'])
        if verbose:
            print("loaded checkpoint '{}'".format(filepath))
        return True
    else:
        if verbose:
            print("no checkpoint found at '{}'".format(filepath))
        return False


def normalize_tensor(t):
    t = t - torch.min(t)
    t = t / torch.max(t)

    return t


def process_imgs(input, recon, first_cam, second_cam, n_factors):
    input = normalize_tensor(input)
    recon = normalize_tensor(recon)

    input = make_grid(input, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().cpu().numpy()
    recon = make_grid(recon, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().cpu().numpy()
    first_cam = make_grid(first_cam, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().cpu().numpy()
    second_cam = make_grid(second_cam, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().cpu().numpy()

    return input, recon, first_cam, second_cam


def add_heatmap(input, gcam):
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + np.asarray(input, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)

    return np.uint8(gcam)


def main(args):
    np.random.seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    model = FactorVAE(args.z_dim).to(device)
    model_found = load_checkpoint(model, args.dir, args.name, device)

    if not model_found:
        return

    gcam = GradCAM(model.encode, args.target_layer, device, args.image_size)

    _, dataset = return_data(args)

    input = dataset[np.arange(0, args.sample_count)][0].to(device)
    recon, mu, logvar, z = model(input)

    input, recon = input.repeat(1, 3, 1, 1), recon.repeat(1, 3, 1, 1)

    maps = gcam.generate(z)
    maps = maps.transpose(0,1)

    first_cam, second_cam = [], []
    for map in maps:
        response = map.flatten(1).sum(1)
        argmax = torch.argmax(response).item()
        first_cam.append(normalize_tensor(map[argmax]))

        response = torch.cat((response[:argmax], response[argmax+1:]))
        second_cam.append(normalize_tensor(map[torch.argmax(response).item()]))

    first_cam = ((torch.stack(first_cam, axis=1)).transpose(0,1)).unsqueeze(1)
    second_cam = ((torch.stack(second_cam, axis=1)).transpose(0,1)).unsqueeze(1)

    input, recon, first_cam, second_cam = process_imgs(input.detach(), recon.detach(), first_cam.detach(), second_cam.detach(), args.sample_count)

    heatmap = add_heatmap(input, first_cam)
    heatmap2 = add_heatmap(input, second_cam)

    input = np.uint8(np.asarray(input, dtype=np.float)*255)
    recon = np.uint8(np.asarray(recon, dtype=np.float)*255)
    grid = np.concatenate((input, heatmap, heatmap2))

    cv2.imshow('Attention Maps of ' + args.name, grid)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizer')

    parser.add_argument('--name', default='main', type=str, help='name of the model to be visualized')
    parser.add_argument('--dir', default='checkpoints', type=str, help='name of the directory holding the models weights')
    parser.add_argument('--output_dir', default='visualizations', type=str, help='name of the directory holding the visualizations')

    parser.add_argument('--seed', default=0, type=int, help='the seed')
    parser.add_argument('--cuda', type=bool, const=True, default=False, nargs='?', help='add if the gpu should be used')

    parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z, necessary for loading the model properly')
    parser.add_argument('--target_layer', type=str, default='0', help='target layer for the attention maps')
    parser.add_argument('--sample_count', default=5, type=int, help='amount of samples from the dataset to create the maps for')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='dsprites', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')
    parser.add_argument('--batch_size', default=1, type=int, help='place holder')

    args = parser.parse_args()

    main(args)
