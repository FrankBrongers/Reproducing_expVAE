from PIL import Image
import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms
import numpy as np

"""
Based on https://github.com/maksimbolonkin/video_anomaly_detection_pytorch/blob/master/ucsd_dataset.py
"""

class UCSDAnomalyDataset(data.Dataset):
    def __init__(self, root_path='./data', train=True, resize=96):
        super(UCSDAnomalyDataset, self).__init__()

        if train:
            self.root_dir = os.path.join(root_path, 'Train')
        else:
            self.root_dir = os.path.join(root_path, 'Test')

        # Get all x folder dirs
        x_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and d[-2:] != 'gt']

        # Get all y folder dirs, None when no y in folder
        y_dirs = []
        for d in x_dirs:
            if os.path.isdir(os.path.join(self.root_dir,(d + '_gt'))):
                y_dirs.append(d + '_gt')
            else:
                y_dirs.append(None)



                
        self.x_samples = []
        self.y_samples = []

        # Construct a list of al possible x's and y's --> None if no target available
        for x_d, y_d in zip(x_dirs, y_dirs):
            for i in range(1, 201):
                self.x_samples.append(os.path.join(self.root_dir, x_d, '{0:03d}.tif'.format((i))))

                # If available add x mask dir
                if y_d:
                    self.y_samples.append(os.path.join(self.root_dir, y_d, '{0:03d}.bmp'.format((i))))
                else:
                    self.y_samples.append(None)


        # uncomment to print counts
        # c = 0        
        # nc = 0
        # for d in self.y_samples:
        #     if d:
        #         c += 1
        #     else:
                
        #         nc += 1
        # print(c, nc)

        # self.pil_transform = transforms.Compose([
        #             transforms.Resize((resize, resize)),
        #             transforms.Grayscale(),
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean=(0.3750352255196134,), std=(0.20129592430286292,))]
                    # )

        self.pil_transform = transforms.Compose([
                    transforms.Resize((resize, resize)),
                    transforms.Grayscale(),
                    transforms.ToTensor()])


    def __getitem__(self, index):
        # Get x
        with open(self.x_samples[index], 'rb') as file:
            x_frame = Image.open(file).convert('RGB')
            x_frame = self.pil_transform(x_frame)

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


# Test main
if __name__ == "__main__":
    dataset = UCSDAnomalyDataset('data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/', train=False)
    print(f"Dataset length: {dataset.__len__()}")

    labels = 0

    for i in range(0, 7199):
        I = dataset.__getitem__(i)

        if I[1].max() > 0.5:
            labels += 1


        if i % 200 == 0:
            print(I[1].unique())
            print(i, I[0].shape, I[1].shape)

    print(labels)
