import argparse
import torch
from torchvision import datasets, transforms

import os
import numpy as np

from models.vanilla import ConvVAE
from models.resnet18 import ResNet18VAE

import OneClassMnist
from gradcam import GradCAM
import cv2
from PIL import Image
from torchvision.utils import save_image, make_grid


def save_cam(image, filename, gcam):
    """
    Saves the attention maps generated by the model.
    Inputs:
        image - original image
        filename - name of to be saved file
        gcam - generated attention map of image
    """
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    cv2.imwrite(filename, gcam)

def main(args):
    """
    Main Function for testing and saving attention maps.
    Inputs:
        args - Namespace object from the argument parser
    """

    torch.manual_seed(args.seed)

    # Load the dataset
    one_class = args.one_class # Choose the current outlier digit to be 8
    one_mnist_test_dataset = OneClassMnist.OneMNIST('./data', one_class, train=False, transform=transforms.ToTensor())

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device == "cuda" else {}
    test_loader = torch.utils.data.DataLoader(
        one_mnist_test_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)


    # Select a model architecture
    if args.model == 'vanilla':
        model = ConvVAE(args.latent_size).to(device)
        target_layer = 'encoder.2'
    elif args.model == 'resnet18':
        model = ResNet18VAE(args.latent_size).to(device)
        # TODO Understand why to choose a specific target layer
        target_layer = 'encoder.layer4.1.conv2'


    # Load model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    mu_avg, logvar_avg = 0, 1
    gcam = GradCAM(model, target_layer=target_layer, device= device)
    test_index=0


    # Generate attention maps
    for batch_idx, (x, _) in enumerate(test_loader):
        model.eval()
        x = x.to(device)
        x_rec, mu, logvar = gcam.forward(x)

        model.zero_grad()
        gcam.backward(mu, logvar, mu_avg, logvar_avg)
        gcam_map = gcam.generate()
        print("x, heatmap", x.size(), gcam_map.size())
        # Visualize and save attention maps
        x = x.repeat(1, 3, 1, 1)
        for i in range(x.size(0)):
            raw_image = x[i] * 255.0
            ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
            im = Image.fromarray(ndarr.astype(np.uint8))
            im_path = args.result_dir
            if not os.path.exists(im_path):
                os.mkdir(im_path)
            im.save(os.path.join(im_path,
                             "{}-{}-origin.png".format(test_index, str(one_class))))

            file_path = os.path.join(im_path,
                                 "{}-{}-attmap.png".format(test_index, str(one_class)))
            r_im = np.asarray(im)
            save_cam(r_im, file_path, gcam_map[i].squeeze().cpu().data.numpy())
            test_index += 1

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Explainable VAE')
    parser.add_argument('--result_dir', type=str, default='test_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.')

    # model option
    parser.add_argument('--model', type=str, default='vanilla',
                        help='select one of the following models: vanilla, resnet18')
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--model_path', type=str, default='./ckpt/vanilla_best.pth', metavar='DIR',
                        help='pretrained model directory')
    parser.add_argument('--one_class', type=int, default=7, metavar='N',
                        help='inlier digit for one-class VAE training')

    args = parser.parse_args()

    main(args)
