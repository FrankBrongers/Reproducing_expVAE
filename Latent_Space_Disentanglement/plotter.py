import argparse, os, json
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


def main(args):
    """
    Generate the visualizations of the results.
    """
    names = args.names.split()

    all_files = os.listdir(args.dir)
    files = [file for file in all_files if file[:-7] in names]

    assert len(files) > 0, 'No files found'

    data = defaultdict(list)
    for file in files:
        data[file[:-7]].append(load_data(os.path.join(args.dir, file)))

    variables = ['vae_recon_loss', 'vae_kld', 'vae_tc_loss', 'D_tc_loss', 'dis_score']
    full_variables = ['reconstruction error', 'KL-divergence', 'VAE TC-Loss', 'discriminator TC-loss', 'disentanglement metric']

    # Initial runs had no ad_loss value
    if args.ad_loss:
        variables.append('ad_loss')
        full_variables.append('attention disentanglement loss')

    # Generate plots against iteration
    for i, v in enumerate(variables):
        if not args.all_plots:
            break
        fig = plt.figure()
        for name in names:
            if name not in data.keys():
                continue

            outputs = [file['outputs'][v] for file in data[name]]
            iteration = data[name][0]['outputs']['iteration']
            mean = np.mean(outputs, axis=0)
            std = np.std(outputs, axis=0)

            # Only the runs with the ad_loss flag will have a lambda value
            try:
                assert data[name][0]['args']['ad_loss'] == True
                label = r'$\gamma=$'+str(data[name][0]['args']['gamma']) + \
                        r', $\lambda=$'+str(int(data[name][0]['args']['lamb'])) + \
                        ', L'+str(data[name][0]['args']['target_layer'])
            except:
                label = r'$\gamma=$'+str(data[name][0]['args']['gamma'])


            plt.errorbar(iteration, mean, yerr=std, label=label)

        plt.xlabel('Iterations')
        plt.ylabel(full_variables[i])
        plt.legend()
        plt.show()

    # Generate reconstrucion error against disentanglement score as in the paper
    recons, scores, annotations = [], [], []

    fig = plt.figure()
    for name in names:
        if name not in data.keys():
            print(name, 'not found.')
            continue

        recon = np.mean([file['outputs']['vae_recon_loss'] for file in data[name]], axis=0)[-1]
        dis = np.mean([file['outputs']['dis_score'] for file in data[name]], axis=0)[-1]

        try:
            # Get plotting specifics for AD-FactorVAE
            assert data[name][0]['args']['ad_loss'] == True
            label = 'AD-FactorVAE'
            marker = 'o'
            color = 'red'
            annotation = r'$\lambda$'+str(int(data[name][0]['args']['lamb']))
            if int(data[name][0]['args']['target_layer']) != 0:
                annotation += 'L'+str(data[name][0]['args']['target_layer'])
            recons.append(recon)
            scores.append(dis)
            annotations.append(annotation)
        except:
            # Get plotting specifics for FactorVAE
            label = 'FactorVAE'
            marker = 'D'
            color = 'green'

        plt.scatter(recon, dis, label=label, c=color, marker=marker, edgecolors='black')

    plt.axis([0, 150, .75, 1.00])
    plt.xticks([0, 50, 100, 150])

    plt.xlabel('reconstrucion error')
    plt.ylabel('disentanglement metric')

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, color='w')
    ax.set_facecolor('lightblue')

    for i, annotation in enumerate(annotations):
        ax.annotate(annotation, (recons[i]+1.5, scores[i]+0.003))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()


def load_data(path):
    with open(path) as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score/Loss Plotter')

    parser.add_argument('--names', default='main', type=str, help='names of the experiments to be plotted')
    parser.add_argument('--dir', default='results', type=str, help='name of the directory holding the results')
    parser.add_argument('--ad_loss', type=bool, const=True, default=False, nargs='?', help='add if the attention disentanglement loss should be used')
    parser.add_argument('--all_plots', type=bool, const=True, default=False, nargs='?', help='add if scores and losses should be plotted against iterations')

    args = parser.parse_args()

    main(args)
