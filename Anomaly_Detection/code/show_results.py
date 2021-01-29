import matplotlib
import argparse
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend("agg")
def main(args):
    im_path = 'test_results/{}_{}_{}'.format(args.dataset, args.model, args.one_class)
    print(args.dataset, "data")
    if args.dataset == "mnist":
        batch_max = 31
        idx_max = 31
    if args.dataset == "mvtec_ad":
        batch_max = 40
        idx_max = 1
    for i in range(2):
        batch_idx = random.randint(0, batch_max )
        i = random.randint(0, idx_max)
        img_orig = mpimg.imread(os.path.join(im_path, "{}-{}-origin.png".format(batch_idx, i)))
        img_att = mpimg.imread(os.path.join(im_path, "{}-{}-attmap.png".format(batch_idx, i)))
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(img_orig)
        f.add_subplot(1,2, 2)
        plt.imshow(img_att)
        plt.show(block=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score/Loss Plotter')

    parser.add_argument('--dataset', type=str, default='ucsd_ped1',
                        help='select one of the following datasets: mnist, ucsd_ped1, mvtec_ad')
    # parser.add_argument('--dir', default='results', type=str, help='name of the directory holding the results')
    # parser.add_argument('--ad_loss', type=bool, const=True, default=False, nargs='?', help='add if the attention disentanglement loss should be used')
    # parser.add_argument('--all_plots', type=bool, const=True, default=False, nargs='?', help='add if scores and losses should be plotted against iterations')
    parser.add_argument('--model', type=str, default='vanilla_ped1',
                    help='select one of the following models: vanilla_mnist, vanilla_ped1, resnet18')
    parser.add_argument('--one_class', type=int, default=7, metavar='N',
                        help='inlier digit for one-class VAE training')
    args = parser.parse_args()

    main(args)
