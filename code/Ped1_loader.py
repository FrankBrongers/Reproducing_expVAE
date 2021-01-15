from PIL import Image
import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms

URL = 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz'


class UCSDAnomalyDataset(data.Dataset):
    def __init__(self, root_path='./data', train=True, resize=100):
        super(UCSDAnomalyDataset, self).__init__()

        if train:
            self.root_dir = os.path.join(root_path, 'Train')
        else:
            self.root_dir = os.path.join(root_path, 'Test')

        # Get all image dirs
        dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and d[-2:] != 'gt']
        self.samples = []

        for d in dirs:
            for i in range(1, 201):
                self.samples.append(os.path.join(self.root_dir, d, '{0:03d}.tif'.format((i))))

        self.pil_transform = transforms.Compose([
                    transforms.Resize((resize, resize)),
                    transforms.Grayscale(),
                    transforms.ToTensor()])
        # self.tensor_transform = transforms.Compose([
        #             transforms.Normalize(mean=(0.3750352255196134,), std=(0.20129592430286292,))])
        
    def __getitem__(self, index):
        with open(self.samples[index], 'rb') as fin:
            frame = Image.open(fin).convert('RGB')
            frame = self.pil_transform(frame) / 255.0
        return frame, torch.zeros_like(frame) # TODO return gt


    def __len__(self):
        return len(self.samples)


# Test main
if __name__ == "__main__":
    dataset = UCSDAnomalyDataset('data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/', train=False)
    print(f"Dataset length: {dataset.__len__()}")

    for i in range(0, 7199):
        I = dataset.__getitem__(i)
        if i % 1000 == 0:
            print(i, I[0].shape)

