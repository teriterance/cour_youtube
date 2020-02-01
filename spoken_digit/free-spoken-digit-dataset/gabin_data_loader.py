import torch
import os 
import pandas as pd 
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class GabinSpectrogramDataset(Dataset):
    def __init__(self, root_dir, status = "train",transforms = None):
        """transformation a appliquer au dataset, et dossier de ce dernier 
        :root_dir =  dossier du dataset 
        :status = 
        :transform = transformation a apliquer sur chaque element  
        """
        self.root_dir = root_dir

        if status == "test":
            self.status = "testing"
        elif status == "train":
            self.status = "training"
        self.file_dir = self.root_dir+ "/"+self.status+ "-spectrograms"
        self.transform = transforms
        #remplissage d'une liste d'element du datasset
        self.files = []
        for ( _ , _ , filenames) in os.walk(self.file_dir):
            self.files.extend(filenames)
    
    def __len__(self):
        """renvoi la taille du datasset"""
        return len(self.files)

    def __getitem__(self, idx):
        """renvoi l'un element
        :idx = indice de l'element
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.file_dir, self.files[idx])
        image = io.imread(img_name)
        
        sample = {'image': image, 'cat': int(self.files[idx][0])}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, cat = sample['image'], sample['cat']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'cat': torch.tensor(cat)}


if __name__ == "__main__":
    gabinDataset  =  GabinSpectrogramDataset(".")
    image = gabinDataset[0]
    print(image["image"].shape, image["cat"])
    dataloader = DataLoader(gabinDataset, batch_size=4,shuffle=True, num_workers=4)

    """
    for i in range(len(gabinDataset)):
        sample = gabinDataset[i]

        print(i, sample['image'].shape, sample['cat'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['image'])
        if i == 3:
            plt.show()
            break
        """