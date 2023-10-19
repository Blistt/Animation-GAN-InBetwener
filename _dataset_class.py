from torch.utils.data import Dataset
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None, binarize_at=0.0, resize_to=(0,0), crop_shape=(0,0), augment=0.0):
        self.data = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
        self.transform = transform
        self.binarize_at = binarize_at
        self.resize_to = resize_to
        self.crop_shape = crop_shape
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triplet_path = self.data[idx]
        extension = os.listdir(self.data[idx])[0].split(".")[-1]
        #Extracts all three frames of the triplet
        frame1 = self.transform(Image.open(triplet_path + "/frame1." + extension))
        frame2 = self.transform(Image.open(triplet_path + "/frame2." + extension))
        frame3 = self.transform(Image.open(triplet_path + "/frame3." + extension))

        # DATA AUGMENTATION
        if self.augment > 0.0:
            # Randomly apply horizontal flip
            if np.random.rand() > self.augment:
                frame1 = transforms.functional.hflip(frame1)
                frame2 = transforms.functional.hflip(frame2)
                frame3 = transforms.functional.hflip(frame3) 
            # Randomly reverse the order of the frames
            if np.random.rand() > self.augment:
                frame1, frame3 = frame3, frame1
            

        if self.binarize_at > 0.0:            
            frame1 = (frame1 > self.binarize_at).float()
            frame2 = (frame2 > self.binarize_at).float()
            frame3 = (frame3 > self.binarize_at).float()
            
        if self.crop_shape != (0,0):
            frame2 = transforms.functional.center_crop(frame2, self.crop_shape)
    
        return frame1, frame2, frame3