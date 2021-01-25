import argparse
import torch
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc

import os
import numpy as np
import matplotlib.pyplot as plt

from models.vanilla import ConvVAE
from models.vanilla_ped1 import ConvVAE_ped1
from models.resnet18 import ResNet18VAE
from models.resnet18_enc_only import ResNet18VAE_2

import OneClassMnist
import Ped1_loader
import MVTec_loader as mvtec

from gradcam import GradCAM
import cv2
from PIL import Image
from torchvision.utils import save_image, make_grid

# Initialize AUROC parameters
test_steps = 100 # Choose a very high number to test the whole dataset
score_range = 50 # How many threshold do you want to test?
scores = np.zeros((score_range, 4)) # TP, TN, FP, FN
plot_ROC = False # Plot the ROC curve or not

save_gcam_image = True
norm_gcam_image = True

def save_cam(image, filename, gcam):
    """
    Saves the attention maps generated by the model.
    Inputs:
        image - original image
        filename - name of to be saved file
        gcam - generated attention map of image
    """
    # Normalize
    if norm_gcam_image:
        gcam = gcam - np.min(gcam)
        gcam = gcam / np.max(gcam)
    else:
        # Divide by a hand-chosen maximum value
        gcam = gcam / 1.5
        gcam = np.clip(gcam, 0.0, 1.0)

    # Save image
    if save_gcam_image:
        h, w, d = image.shape

        save_gcam = cv2.resize(gcam, (w, h))
        save_gcam = cv2.applyColorMap(np.uint8(255 * save_gcam), cv2.COLORMAP_JET)
        save_gcam = np.asarray(save_gcam, dtype=np.float) + \
            np.asarray(image, dtype=np.float)
        save_gcam = 255 * save_gcam / np.max(save_gcam) # With norm
        # print(np.unique(save_gcam), save_gcam.min(), save_gcam.max())
        save_gcam = np.uint8(save_gcam)
        cv2.imwrite(filename, save_gcam) # Uncomment to save the images
    return gcam

def main(args):
    """
    Main Function for testing and saving attention maps.
    Inputs:
        args - Namespace object from the argument parser
    """

    torch.manual_seed(args.seed)

    # Load the dataset
    one_class = args.one_class # Choose the current outlier digit to be 8

    # Load dataset
    if args.dataset == 'mnist':
        test_dataset = OneClassMnist.OneMNIST('./data', args.one_class, train=False, transform=transforms.ToTensor())
    elif args.dataset == 'ucsd_ped1':
        test_dataset = Ped1_loader.UCSDAnomalyDataset('data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/', train=False, resize=100)
    elif args.dataset == 'mvtec_ad':
        # for dataloader check: pin pin_memory, batch size 32 in original
        class_name = mvtec.CLASS_NAMES[5]
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False, grayscale=False)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device == "cuda" else {}
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Select a model architecture
    if args.model == 'vanilla':
        model = ConvVAE(args.latent_size).to(device)
    elif args.model == 'vanilla_ped1':
        model = ConvVAE_ped1(args.latent_size).to(device)
    elif args.model == 'resnet18':
        model = ResNet18VAE(args.latent_size).to(device)
        # TODO Understand why to choose a specific target layer
    elif args.model == 'resnet18_2':
        model = ResNet18VAE_2(args.latent_size, x_dim =256, nc = 3).to(device)
        # TODO Understand why to choose a specific target layer
    print("layer issss", args.target_layer)
    # Load model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    mu_avg, logvar_avg = 0, 1
    gcam = GradCAM(model, target_layer=args.target_layer, device=device)
    test_index=0

    # Generate attention maps
    for batch_idx, (x, y) in enumerate(test_loader):
        # print("batch_idx", batch_idx)
        model.eval()
        x = x.to(device)
        x_rec, mu, logvar = gcam.forward(x)

        model.zero_grad()
        gcam.backward(mu, logvar, mu_avg, logvar_avg)
        gcam_map = gcam.generate()

        # If image has one channel, make it three channel(need for heatmap)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Visualize and save attention maps
        for i in range(x.size(0)):
            # for every image in batch

            raw_image = x[i] * 255.0

            # ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()[:,:,:3]
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
            pred = gcam_map[i].squeeze().cpu().data.numpy()
            pred = save_cam(r_im, file_path, pred)

            # Compute the correct and incorrect mask scores for all thresholds
            for j, score in enumerate(scores):

                threshold = (j + 1) / score_range

                # Apply the threshold
                pred_bin = ((pred) > threshold).astype(int)
                gt_mask = y[i,:,:,:].numpy().astype(int)

                TP = np.sum((pred_bin + gt_mask) == 2)
                TN = np.sum((pred_bin + gt_mask) == 0)

                FP = np.sum((gt_mask - pred_bin) == -1)
                FN = np.sum((pred_bin - gt_mask) == -1)
                # print(np.array([TP, TN, FP, FN]))
                scores[j] += np.array([TP, TN, FP, FN])
            test_index += 1

        # Stop parameter
        if batch_idx == test_steps:
            print("Reached the maximum number of steps")
            break

    # Compute AUROC
    TPR_list = []
    FPR_list = []
    half_list_x = []
    half_list_y = []
    best_threshold_idx = 0

    for i, score in enumerate(scores):
        TP, TN, FP, FN = (score[0], score[1], score[2], score[3])

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPR_list.append(TPR)
        FPR_list.append(FPR)

        half_list_x.append(i / score_range)
        half_list_y.append(i / score_range)

        # Check if current threshold is the best
        if (TPR - FPR) > (TPR_list[best_threshold_idx] - FPR_list[best_threshold_idx]):
            best_threshold_idx = i

    print(f"AUC: {auc(FPR_list, TPR_list)} Best threshold: {best_threshold_idx / score_range}")

    if plot_ROC:
        plt.plot(FPR_list, TPR_list, label="ROC")
        plt.plot(half_list_x, half_list_y, '--')
        plt.scatter(FPR_list[best_threshold_idx], TPR_list[best_threshold_idx])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig("./test_results/auroc_" + str(args.target_layer)+ ".png")
        # plt.show()
    return


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device", device)

    parser = argparse.ArgumentParser(description='Explainable VAE')
    parser.add_argument('--result_dir', type=str, default='test_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to use in the data loaders.')

    # model option
    parser.add_argument('--model', type=str, default='vanilla_ped1',
                        help='select one of the following models: vanilla, vanilla_ped1, resnet18')
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--model_path', type=str, default='./ckpt/vanilla_ped1_checkpoint.pth', metavar='DIR',
                        help='pretrained model directory')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ucsd_ped1',
                        help='select one of the following datasets: mnist, ucsd_ped1, mvtec_ad')
    parser.add_argument('--one_class', type=int, default=7, metavar='N',
                        help='inlier digit for one-class VAE training')

    # AUROC parameters
    parser.add_argument('--target_layer', type=str, default='encoder.4',
                        help='select a target layer for generating the attention map.')

    args = parser.parse_args()

    main(args)
