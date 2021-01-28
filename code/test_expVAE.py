import argparse
import torch
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, roc_auc_score

import os
import numpy as np
import matplotlib.pyplot as plt

from models.vanilla import ConvVAE
from models.vanilla_ped1 import ConvVAE_ped1
from models.resnet18 import ResNet18VAE
from models.resnet18_2 import ResNet18VAE_2
from models.resnet18_3 import ResNet18VAE_3


import OneClassMnist
import Ped1_loader
import MVTec_loader as mvtec

from gradcam import GradCAM
# import cv2
from PIL import Image
from torchvision.utils import save_image, make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize AUROC parameters
test_steps = 100 # Choose a very high number to test the whole dataset
plot_ROC = True # Plot the ROC curve or not

save_gcam_image = True
# norm_gcam_image = True

def save_gradcam(image, filename, gcam, gcam_max = 1):
    """
    Saves the attention maps generated by the model.
    Inputs:
        image - original image
        filename - name of to be saved file
        gcam - generated attention map of image
    """
    # Normalize
    # if norm_gcam_image:
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    # else:
    #     # Divide by a hand-chosen maximum value
    #     gcam = gcam / gcam_max
    #     gcam = np.clip(gcam, 0.0, 1.0)

    # Save image
    # if save_gcam_image:
    #     h, w, d = image.shape
    #
    #     save_gcam = cv2.resize(gcam, (w, h))
    #     save_gcam = cv2.applyColorMap(np.uint8(255 * save_gcam), cv2.COLORMAP_JET)
    #     save_gcam = np.asarray(save_gcam, dtype=np.float) + np.asarray(image, dtype=np.float)
    #     save_gcam = 255 * save_gcam / np.max(save_gcam) # With norm
    #     save_gcam = np.uint8(save_gcam)
    #     cv2.imwrite(filename, save_gcam) # Uncomment to save the images
    return

def main(args):
    """
    Main Function for testing and saving attention maps.
    Inputs:
        args - Namespace object from the argument parser
    """

    torch.manual_seed(args.seed)

    # Load dataset
    if args.dataset == 'mnist':
        test_dataset = OneClassMnist.OneMNIST('./data', args.one_class, train=False, transform=transforms.ToTensor())
    elif args.dataset == 'ucsd_ped1':
        test_dataset = Ped1_loader.UCSDAnomalyDataset('./data', train=False, resize=args.image_size)
    elif args.dataset == 'mvtec_ad':
        # for dataloader check: pin pin_memory, batch size 32 in original

        class_name = mvtec.CLASS_NAMES[args.one_class]
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False, grayscale=False)
        test_steps = len(test_dataset)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device == "cuda" else {}
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Select a model architecture
    if args.model == 'vanilla':
        model = ConvVAE(args.latent_size).to(device)
    elif args.model == 'vanilla_ped1':
        imshape = [1, 192, 144, 96]
        model = ConvVAE_ped1(args.latent_size, args.image_size, imshape).to(device)
    elif args.model == 'resnet18':
        model = ResNet18VAE(args.latent_size).to(device)
        # TODO Understand why to choose a specific target layer
    elif args.model == 'resnet18_2':
        model = ResNet18VAE_2(args.latent_size, x_dim=256, nc=3).to(device)
        # TODO Understand why to choose a specific target layer
    elif args.model == 'resnet18_3':
        imshape = [64, 3, 256, 256 ]
        model = ResNet18VAE_3(args.latent_size, x_dim = imshape[-1], nc = imshape[1]).to(device)
    print("layer issss", args.target_layer)
    # Load model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    mu_avg, logvar_avg = (0, 1)
    gcam = GradCAM(model, target_layer=args.target_layer, device=device)
    test_index = 0

    prediction_stack = np.zeros((test_steps * args.batch_size, imshape[-1], imshape[-1]), dtype=np.float32)
    gt_mask_stack = np.zeros((test_steps * args.batch_size, imshape[-1], imshape[-1]), dtype=np.uint8)


    # Generate attention maps
    for batch_idx, (x, y) in enumerate(test_loader):

        # print("batch_idx", batch_idx)
        model.eval()
        x = x.to(device)
        x_rec, mu, logvar = gcam.forward(x)

        model.zero_grad()
        gcam.backward(mu, logvar, mu_avg, logvar_avg)
        gcam_map = gcam.generate()
        gcam_max = torch.max(gcam_map).item()

        # Unnormalize for saving
        # x = test_dataset.unnormalize(x)

        # If image has one channel, make it three channel(need for heatmap)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Visualize and save attention maps
        for i in range(x.size(0)):
            x_arr = x[i].permute(1, 2, 0).cpu().numpy() * 255
            x_im = Image.fromarray(x_arr.astype(np.uint8))

            if save_gcam_image:
                im_path = args.result_dir
                if not os.path.exists(im_path):
                    os.mkdir(im_path)
                x_im.save(os.path.join(im_path, "{}-{}-origin.png".format(batch_idx, i)))
                file_path = os.path.join(im_path, "{}-{}-attmap.png".format(batch_idx, i))

            # Get the gradcam for this image
            prediction = gcam_map[i].squeeze().cpu().data.numpy()
            save_gradcam(x_arr, file_path, prediction, gcam_max = gcam_max)

            # Add prediction and mask to the stacks
            prediction_stack[batch_idx*args.batch_size + i] = prediction
            gt_mask_stack[batch_idx*args.batch_size + i] = y[i]

        # Stop parameter
        if batch_idx == (test_steps - 1):
            print("Reached the maximum number of steps")
            break

    # Compute area under the ROC score
    auc = roc_auc_score(gt_mask_stack.flatten(), prediction_stack.flatten())
    print(f"AUROC score: {auc}")

    if plot_ROC:
        tpr, tnr, _ =  roc_curve(gt_mask_stack.flatten(), prediction_stack.flatten())
        plt.plot(tpr, tnr, label="ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig("./test_results/auroc_" + str(args.target_layer)+ str(args.one_class)+ ".png")
    return


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device", device)

    parser = argparse.ArgumentParser(description='Explainable VAE')
    parser.add_argument('--result_dir', type=str, default=None, metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to use in the data loaders.')

    # Model option
    parser.add_argument('--model', type=str, default='vanilla_ped1',
                        help='select one of the following models: vanilla, vanilla_ped1, resnet18')
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--model_path', type=str, default='/media/bob/OS/Users/boble/Documents/AI - year 1/FACT-AI/vanilla_ped1_best_120_deeper.pth', metavar='DIR',
                        help='pretrained model directory')
    parser.add_argument('--image_size', type=int, default=96,
                        help='Select an image size')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ucsd_ped1',
                        help='select one of the following datasets: mnist, ucsd_ped1, mvtec_ad')
    parser.add_argument('--one_class', type=int, default=7, metavar='N',
                        help='inlier digit for one-class VAE training')

    # AUROC parameters
    parser.add_argument('--target_layer', type=str, default='encoder.8',
                        help='select a target layer for generating the attention map.')
    parser.add_argument('--decoder', type=str, default='vanilla',
                        help='only for resnet VAE select one of following: resnet, vanilla')
    parser.add_argument('--no_auroc', default=True, action='store_false',
                        help='if this argument is passed, the auroc score will not be computed')

    args = parser.parse_args()

    # If no argument for result directory is specified, set it to data and model name
    if args.result_dir is None:
        args.result_dir = 'test_results/{}_{}_{}'.format(args.dataset, args.model, args.one_class)

    main(args)
