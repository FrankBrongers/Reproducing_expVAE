import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import os
import shutil
import numpy as np

from models.vanilla import ConvVAE
# from models.resnet18 import ...

import OneClassMnist


def loss_function(recon_x, x, mu, logvar):
    """
    Calculates the reconstruction (binary cross entropy) and regularization (KLD) losses to form the total loss of the VAE.
    Inputs:
        recon_x - Batch of reconstructed images of shape [B,C,H,W].
        x - Batch of original input images of shape [B,C,H,W].
        mu - Mean of the posterior distributions.
        log_var - Log standard deviation of the posterior distributions.
    """    
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)

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

    test_loss /= len(test_loader.dataset)

    return test_loss


def save_checkpoint(state, is_best, outdir):
    """
    Function for saving pytorch model checkpoints.
    Inputs:
        state - 
        is_best - VAE model to train
        outdir - Data Loader for the dataset you want to train on
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, f"{state['model']}_checkpoint.pth")
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

    # Seed everything
    torch.manual_seed(args.seed)


    # Load datasets
    one_class = args.one_class # Choose the inlier digit to be 3
    one_mnist_train_dataset = OneClassMnist.OneMNIST('./data', one_class, train=True, download=True, transform=transforms.ToTensor())
    one_mnist_test_dataset = OneClassMnist.OneMNIST('./data', one_class, train=False, transform=transforms.ToTensor())
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        one_mnist_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        one_mnist_test_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)


    # Select a model architecture
    if args.model == 'vanilla':
        model = ConvVAE(args.latent_size).to(device)
    elif args.model == 'resnet18':
        model = TODO


    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    start_epoch = 0
    best_test_loss = np.finfo('f').max

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)


    # Training and testing
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(model, train_loader, optimizer, args)
        test_loss = test(model, test_loader,args) 

        print('Epoch [%d/%d] loss: %.3f val_loss: %.3f' % (epoch + 1, args.epochs, train_loss, test_loss))

        # Check if model is good enough for checkpoint to be created
        is_best = test_loss < best_test_loss
        best_test_loss = min(test_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model' : args.model,
        }, is_best, os.path.join('./',args.ckpt_dir))

        # Visualize sample validation result
        with torch.no_grad():
            sample = torch.randn(64, 32).to(device)
            sample = model.decode(sample).cpu()
            img = make_grid(sample)
            save_dir = os.path.join('./',args.result_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(sample.view(64, 1, 28, 28), os.path.join(save_dir,'sample_' + str(epoch) + '.png'))


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
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.')


    # Model parameters
    parser.add_argument('--model', type=str, default='vanilla',
                        help='select one of the following models: vanilla, resnet18')
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--one_class', type=int, default=3, metavar='N',
                        help='inlier digit for one-class VAE training')
    

    args = parser.parse_args()

    main(args)