import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

import os
import shutil
import numpy as np

from models.vanilla_mnist import ConvVAE_mnist
from models.vanilla_ped1 import ConvVAE_ped1
from models.resnet18 import ResNet18VAE
from models.resnet18_2 import ResNet18VAE_2
from models.resnet18_3 import ResNet18VAE_3


import OneClassMnist
import Ped1_loader
import MVTec_loader as mvtec

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
# Run the folloring command to acces tensorboard: tensorboard --logdir runs
# from torchvision import transforms as T


def loss_function(recon_x, x, mu, logvar, ):
    """
    Calculates the reconstruction (binary cross entropy) and regularization (KLD) losses to form the total loss of the VAE.
    Inputs:
        recon_x - Batch of reconstructed images of shape [B,C,H,W].
        x - Batch of original input images of shape [B,C,H,W].
        mu - Mean of the posterior distributions.
        log_var - Log standard deviation of the posterior distributions.
    """
    B = recon_x.shape[0]
    rc = recon_x.shape[1]
    # if rc == 1:
    if True:
        rec_loss = F.binary_cross_entropy(recon_x.view(B, -1), x.view(B, -1), reduction='sum').div(B)
    else:
        rec_loss = F.mse_loss(x, recon_x, reduction = 'sum').div(B)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).div(B)

    return rec_loss + KLD


def train(model, train_loader, optimizer, args):
    """
    Function for training a model on a dataset. Train for one epoch.
    Inputs:
        model - VAE model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        train_loss - Averaged total loss
    """
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)/args.batch_size
    return train_loss


def test(model, test_loader, args):
    """
    Function for testing the model on a test dataset. Test for one epoch.
    Inputs:
        model - VAE model to train
        test_loader - Data Loader for the dataset you want to test on
    Outputs:
        test_loss - Averaged total loss on test set
    """
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)/args.batch_size
    return test_loss


def save_checkpoint(state, is_best, outdir, args):
    """
    Function for saving pytorch model checkpoints.
    Inputs:
        state - state of the model containing current epoch, model, optimizer and loss
        is_best - boolean stating if model has best test loss so far
        outdir - directory to save checkpoints
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if args.dataset == "mvtec_ad":
            checkpoint_file = os.path.join(outdir, f"{state['model']}_mvtecClass_"+str(args.one_class) +"_checkpoint.pth")
            best_file = os.path.join(outdir, f"{state['model']}_mvtecClass_"+str(args.one_class)+"_best.pth")
    else:
        checkpoint_file = os.path.join(outdir, f"{state['model']}_"+str(args.decoder) +"_checkpoint.pth")
        best_file = os.path.join(outdir, f"{state['model']}_best.pth")
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

def main(args):
    """
    Main Function for the full training & evaluation loop of the VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """
    print("Device is", device)

    # Seed everything
    torch.manual_seed(args.seed)

    # Load dataset
    if args.dataset == 'mnist':
        # for generating images
        imshape = [64, 1, args.image_size, args.image_size]
        one_class = args.one_class # Choose the inlier digit to be 3
        train_dataset = OneClassMnist.OneMNIST('./data', one_class, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = OneClassMnist.OneMNIST('./data', one_class, train=False, transform=transforms.ToTensor())
    elif args.dataset == 'ucsd_ped1':
        imshape = [64, 1, args.image_size, args.image_size]
        train_dataset = Ped1_loader.UCSDAnomalyDataset('./data/', train=True, resize=args.image_size)
        test_dataset = Ped1_loader.UCSDAnomalyDataset('./data', train=False, resize=args.image_size)
    elif args.dataset == 'mvtec_ad':
        # for dataloader check: pin pin_memory, batch size 32 in original
        imshape = [64, 3, 256, 256 ]
        class_name = mvtec.CLASS_NAMES[args.one_class]   # nuts
        train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True, grayscale=False)


    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Select a model architecture
    if args.model == 'vanilla':
        model = ConvVAE_mnist(args.latent_size).to(device)
    elif args.model == 'vanilla_ped1':
        model = ConvVAE_ped1(args.latent_size, args.image_size, [1, 64, 128, 256], batch_norm=False).to(device)
    elif args.model == 'resnet18':
        model = ResNet18VAE(args.latent_size, x_dim = imshape[-1], nc = imshape[1]).to(device)
    elif args.model == 'resnet18_2':
        model = ResNet18VAE_2(args.latent_size, x_dim = imshape[-1], nc = imshape[1], decoder=args.decoder).to(device)
    elif args.model == 'resnet18_3':
        model = ResNet18VAE_3(args.latent_size, x_dim = imshape[-1], nc = imshape[1]).to(device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)

    start_epoch = 0
    best_train_loss = np.finfo('f').max
    losses = []
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_train_loss = checkpoint['best_train_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)


    # Training and testing
    for epoch in range(start_epoch, args.epochs):
        if args.vae_testsave == True:
            with torch.no_grad():
                save_dir = os.path.join('./',args.result_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                testim =  next(iter(train_loader))[0][0][None,:].to(device)
                gen_testim = model(testim)[0]

                combi = make_grid([testim[0].cpu(), gen_testim[0].cpu()], padding=100)
                save_image(combi.cpu(), os.path.join(save_dir,str(args.decoder) +"combi_"+ str(epoch) + '.png'))

        train_loss = train(model, train_loader, optimizer, args)
        # test_loss = test(model, test_loader,args)

        # writer.add_scalar('Train Loss', train_loss, epoch)
        # writer.add_scalar('Test Loss', test_loss, epoch)

        print('Epoch [%d/%d] loss: %.3f ' % (epoch + 1, args.epochs, train_loss))
        print(f"Lr: {optimizer.param_groups[0]['lr']}")

        # save trainloss plot
        losses.append(train_loss)
        plt.plot(losses)
        plt.savefig("./train_results/loss_" + str(args.decoder)+ ".png")

        # Check if model is good enough for checkpoint to be created
        is_best = train_loss < best_train_loss
        best_train_loss = min(train_loss, best_train_loss)
        save_checkpoint({
            'epoch': epoch,
            'best_train_loss': best_train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model' : args.model,
        }, is_best, os.path.join('./',args.ckpt_dir), args)

        # Visualize sample validation result
        with torch.no_grad():
            sample = torch.randn(64, args.latent_size).to(device)
            sample = model.decode(sample).cpu()
            save_dir = os.path.join('./',args.result_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(sample.view(imshape), os.path.join(save_dir,str(args.decoder) +'sample_' + str(epoch) + '.png'))
        scheduler.step()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Explainable VAE')

    # Path options
    parser.add_argument('--result_dir', type=str, default='train_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', metavar='DIR',
                        help='ckpt directory')
    parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--epochs', type=int, default=512, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.')

    # Model parameters
    parser.add_argument('--model', type=str, default='vanilla_ped1',
                        help='select one of the following models: vanilla_mnist, resnet18, vanilla_ped1')
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--image_size', type=int, default=100,
                        help='Select an image size')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ucsd_ped1',
                        help='select one of the following datasets: mnist, ucsd_ped1, mvtec_ad')
    parser.add_argument('--one_class', type=int, default=1, metavar='N',
                        help='inlier digit for one-class VAE training')
    parser.add_argument('--vae_testsave', type=bool, default=False,
                        help='save input output image of VAE during training')
    parser.add_argument('--decoder', type=str, default='',
                        help='only for resnet VAE select one of following: resnet, vanilla')

    args = parser.parse_args()

    # If no argument for result directory is specified, set it to data and model name
    if args.result_dir is None:
        args.result_dir = 'train_results/{}_{}'.format(args.dataset, args.model)

    main(args)
