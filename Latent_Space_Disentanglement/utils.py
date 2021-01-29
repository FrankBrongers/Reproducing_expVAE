"""utils.py"""

import os, json
import argparse
import subprocess


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_args_outputs(path, args, outputs):
    data = {'args': vars(args), 'outputs': outputs}
    with open(path+'.json', 'w') as f:
        json.dump(data, f)
