import os
import numpy as np
import glob

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms
import cv2 
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import os
import glob
class MyLidcDataset(Dataset):
    def __init__(self, PATHS,Albumentation=False):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.mask_paths = glob.glob(os.path.join(PATHS,"masks","*"))
        self.mask_paths = [i for i in self.mask_paths if ".png" in i]
        self.image_paths = [i.replace("masks","images") for i in self.mask_paths]
        
        self.albumentation = Albumentation


        self.albu_transformations =  albu.Compose([
            albu.ElasticTransform(alpha=1.1,alpha_affine=0.5,sigma=5,p=0.15),
            albu.HorizontalFlip(p=0.15),
            ToTensorV2()
        ])
        self.transformations = transforms.Compose([transforms.ToTensor()])
    def transform(self, image, mask):
        #Transform to tensor
        # if self.albumentation:
        #     #It is always best to convert the make input to 3 dimensional for albumentation
        #     image = image.reshape(512,512,1)
        #     mask = mask.reshape(512,512,1)
        #     # Without this conversion of datatype, it results in cv2 error. Seems like a bug
        #     mask = mask.astype('uint8')
        #     augmented=  self.albu_transformations(image=image,mask=mask)
        #     image = augmented['image']
        #     mask = augmented['mask']
        #     mask= mask.reshape([1,512,512])
        # else:
        image = self.transformations(image)
        mask = self.transformations(mask)

        image,mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image,mask

    def __getitem__(self, index):
        
        image = cv2.imread(self.image_paths[index])/255
        mask = cv2.imread(self.mask_paths[index])/255
        
        image,mask = self.transform(image,mask)
        return image,mask[0:1]

    def __len__(self):
        return len(self.mask_paths)
        
            











        
