import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import tarfile
from tqdm import tqdm
import urllib.request

from PIL import Image



"""
Based on https://github.com/maksimbolonkin/video_anomaly_detection_pytorch/blob/master/ucsd_dataset.py
"""

URL = 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz'

class UCSDAnomalyDataset(data.Dataset):
    def __init__(self, root_path='./data', train=True, resize=96, download=True):
        super(UCSDAnomalyDataset, self).__init__()

        self.root_path = root_path
        self.dataset_mean = 0.3750352255196134
        self.dataset_std = 0.20129592430286292

        if train:
            self.images_dir = os.path.join(root_path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
        else:
            self.images_dir = os.path.join(root_path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test')

        # download dataset if not exist
        if download:
            self.download()

        # Get all x folder dirs
        x_dirs = [d for d in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, d)) and d[-2:] != 'gt']

        # Get all y folder dirs, None when no y in folder
        y_dirs = []
        for d in x_dirs:
            if os.path.isdir(os.path.join(self.images_dir,(d + '_gt'))):
                y_dirs.append(d + '_gt')
            else:
                y_dirs.append(None)

        self.x_samples = []
        self.y_samples = []

        # Construct a list of al possible x's and y's --> None if no target available
        for x_d, y_d in zip(x_dirs, y_dirs):
            for i in range(1, 201):
                self.x_samples.append(os.path.join(self.images_dir, x_d, '{0:03d}.tif'.format((i))))

                # If available add x mask dir
                if y_d:
                    self.y_samples.append(os.path.join(self.images_dir, y_d, '{0:03d}.bmp'.format((i))))
                else:
                    self.y_samples.append(None)

        self.pil_transform = transforms.Compose([
                    transforms.Resize((resize, resize)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    ])

        self.mean_transform = transforms.Compose([
                    transforms.Normalize(mean=(self.dataset_mean,), std=(self.dataset_std,))
                    ])


    def unnormalize(self, input):
        return input.mul_(self.dataset_std).add_(self.dataset_mean)


    def __getitem__(self, index):
        # Get x
        with open(self.x_samples[index], 'rb') as file:
            x_frame = Image.open(file)
            x_frame = self.pil_transform(x_frame)
            x_frame = self.mean_transform(x_frame)
        # Get y
        if self.y_samples[index]:
            with open(self.y_samples[index], 'rb') as file:
                y_frame = Image.open(file)
                y_frame = self.pil_transform(y_frame)

        else:
            y_frame = torch.zeros_like(x_frame)

        assert x_frame.shape == y_frame.shape, 'x and y shape do not match'
        return x_frame, y_frame


    def __len__(self):
        return len(self.x_samples)


    def download(self):
        """Download dataset if not exist"""

        if not os.path.exists(self.images_dir):
            tar_file_path = self.root_path + 'ucsd_ped1.tar.xz'
            if not os.path.exists(tar_file_path):
                download_url(URL, tar_file_path)
            print('unzip downloaded dataset: %s' % tar_file_path)
            tar = tarfile.open(tar_file_path)
            tar.extractall(self.root_path)
            tar.close()
        return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# Test main
if __name__ == "__main__":
    dataset = UCSDAnomalyDataset('data/', train=False)
    print(f"Dataset length: {dataset.__len__()}")

    for i in range(0, 7199):
        I = dataset.__getitem__(i)
        if i % 500 == 0:
            print(I[0].min(), I[0].max())
            I = dataset.unnormalize(I[0])
            print(I.min(), I.max())
