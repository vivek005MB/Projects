import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pdm
import os


class DogBreedDataset(Dataset):
    
    def __init__(self,img_dir,labels_df,transform=None):
        # initializing
        self.img_dir = img_dir
        self.transform = transform
        self.labels_df = labels_df

    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, index):
        # getting the image name
        image_name = os.path.join(self.img_dir, self.labels_df.id[index]) + ".jpg"
        # loading the image
        image = Image.open(image_name)
        # label
        label = self.labels_df.target[index]
        if self.transform:
            image = self.transform(image)
        else:
            tensor_transform = transforms.ToTensor()
            image = tensor_transform(image)
        return [image, label]
        



        
