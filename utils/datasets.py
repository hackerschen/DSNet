import os

import cv2
import numpy as np
import torch
import torch.utils.data
import random

from albumentations import Compose, Resize

class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, img_ext, mask_ext, transform=None, mode="train"):
        self.img_dir = os.path.join(img_dir, mode)
        self.mask_dir = os.path.join(mask_dir, mode)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        img_ids = os.listdir(self.img_dir)
        self.img_ids = [os.path.splitext(os.path.basename(i))[0] for i in img_ids] 

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        mask.append(cv2.imread(os.path.join(self.mask_dir,
                                            img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.float()
        mask = mask.float()
        mask = mask / 255


        return img, mask, {'img_id': img_id}

class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, img_ext, mask_ext, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = 1
        self.img_ids = os.listdir(self.img_dir)
        self.img_ids = [os.path.splitext(os.path.basename(i))[0] for i in self.img_ids] 
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir,
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.float()
        mask = mask.float() / 255

        return img, mask, {'img_id': img_id}
    
class EnhanceDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, img_ext, mask_ext, transform=None, mode="train"):
        self.img_dir = os.path.join(img_dir, mode)
        self.mask_dir = os.path.join(mask_dir, mode)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        img_ids = os.listdir(self.img_dir)
        self.img_ids = [os.path.splitext(os.path.basename(i))[0] for i in img_ids] 

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        mask.append(cv2.imread(os.path.join(self.mask_dir,
                                            img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.float()
        mask = mask.float()
        mask = mask / 255


        return img, mask, {'img_id': img_id}

class PidDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, img_ext, mask_ext, transform=None, mode="train"):
        self.img_dir = os.path.join(img_dir, mode)
        self.mask_dir = os.path.join(mask_dir, mode)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        img_ids = os.listdir(self.img_dir)
        self.img_ids = [os.path.splitext(os.path.basename(i))[0] for i in img_ids] 

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        mask.append(cv2.imread(os.path.join(self.mask_dir,
                                            img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        val_transform = Compose([
            Resize(512, 512),
        ])
        augmented = val_transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        _, _, edge = self.gen_sample(img, mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.float()
        mask = mask.float()
        mask = mask / 255
        # edge = torch.from_numpy(edge).unsqueeze(0)

        return img, mask, edge, {'img_id': img_id}

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, city=True):

        y_k_size = 6
        x_k_size = 6
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

        return image, label, edge