from PIL import Image
import torch.utils.data as data
import os
import torchvision.transforms as transforms

class UCSDAnomalyDataset(data.Dataset):
    def __init__(self, root_dir):
        super(UCSDAnomalyDataset, self).__init__()
        self.root_dir = root_dir

        # Get all image dirs
        vids = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.samples = []
        for d in vids:
            for i in range(1, 200):
                self.samples.append(os.path.join(self.root_dir, d, '{0:03d}.tif'.format((i) + 1)))

        self.pil_transform = transforms.Compose([
                    transforms.Resize((100, 100)),
                    transforms.Grayscale(),
                    transforms.ToTensor()])
        # self.tensor_transform = transforms.Compose([
                    # transforms.Normalize(mean=(0.3750352255196134,), std=(0.20129592430286292,))])
        
    def __getitem__(self, index):
        # Select folder
        with open(self.samples[index], 'rb') as fin:
            frame = Image.open(fin).convert('RGB')
            frame = self.pil_transform(frame) / 255.0
        # sample = torch.stack(sample, axis=0)
        return frame

    def __len__(self):
        return len(self.samples)

# Test main
if __name__ == "__main__":
    train_ds = UCSDAnomalyDataset('data/Ped1/UCSD_anomaly/UCSDped1/Train')
    train_ds.__getitem__(200)
