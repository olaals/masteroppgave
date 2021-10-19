import os
import sys
sys.path.append("utils")
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import matplotlib.pyplot as plt

def get_dataloaders(batch_size, train_transforms):
    ds = CornerScanV2(transforms=train_transforms)
    ds_len = len(ds)
    train_len = int(ds_len*0.9)
    val_len = ds_len - train_len
    print("ds len", ds_len)
    print("train len", train_len)
    print("val len", val_len)
    train_set, val_set = torch.utils.data.random_split(ds, [train_len,val_len ])
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    return train_loader, val_loader

def get_realscan_v1_loader(batch_size, transforms):
    ds = RealScanV1(transforms=transforms)
    train_loader = DataLoader(ds, batch_size, shuffle=False)
    return train_loader


class RealScanV1(Dataset):
    def __init__(self, transforms=None):
        super().__init__()
        self.transforms = transforms
        imgs_dir = os.path.join("datasets","real-scans-v1")
        self.imgs_paths = [os.path.join(imgs_dir, img_file) for img_file in os.listdir(imgs_dir)]
        self.imgs_paths.sort()

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):

        img = np.array(Image.open(self.imgs_paths[idx]).convert("RGB"))

        if self.transforms:
            augmentations = self.transforms(image=img)
            img = augmentations["image"]

        img = torch.tensor(img, dtype=torch.float32).permute([2,0,1])
        return img

def get_double_scan_v1_loader(batch_size, train_transforms):
    ds = DoubleScanV1(transforms=train_transforms)
    ds_len = len(ds)
    train_len = int(ds_len*0.9)
    val_len = ds_len - train_len
    print("ds len", ds_len)
    print("train len", train_len)
    print("val len", val_len)
    train_set, val_set = torch.utils.data.random_split(ds, [train_len,val_len ])
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    return train_loader, val_loader


class DoubleScanV1(Dataset):
    def __init__(self, transforms=None):
        super().__init__()
        self.transforms = transforms
        dataset_dir = os.path.join("datasets","double-scan-v1")

        self.imgs_paths_l = [os.path.join(dataset_dir, scan_dir, "img_l.png") for scan_dir in os.listdir(dataset_dir)]
        self.imgs_paths_proj = [os.path.join(dataset_dir, scan_dir, "proj.png") for scan_dir in os.listdir(dataset_dir)]
        self.segs_paths = [os.path.join(dataset_dir, scan_dir, "gt_scan.png") for scan_dir in os.listdir(dataset_dir)]
        self.imgs_paths_l.sort()
        self.imgs_paths_proj.sort()
        self.segs_paths.sort()

        self.segs_paths=         self.segs_paths[:120]
        self.imgs_paths_proj=   self.imgs_paths_proj[:120]
        self.imgs_paths_l=      self.imgs_paths_l[:120]
        
        print(len(self.imgs_paths_l))
        print(len(self.segs_paths))
        print(self.imgs_paths_l[:5])

        print(self.segs_paths[:5])

    def __len__(self):
        return len(self.imgs_paths_l)

    def __getitem__(self, idx):

        img_l = np.array(Image.open(self.imgs_paths_l[idx]).convert("RGB"))
        img_l = np.max(img_l, axis=2)
        proj = np.array(Image.open(self.imgs_paths_proj[idx]).convert("RGB"))
        proj = np.max(proj, axis=2)

        img = np.dstack((img_l, proj, np.zeros_like(img_l)))

        seg = np.array(Image.open(self.segs_paths[idx]).convert("L"))
        seg = seg.clip(max=1)

        if self.transforms:
            augmentations = self.transforms(image=img, mask=seg)
            img = augmentations["image"]
            seg = augmentations["mask"]


        img = torch.tensor(img, dtype=torch.float32).permute([2,0,1])
        seg = torch.tensor(seg, dtype=torch.torch.int64)

        return img,seg


    def get_np_img(self, idx):
        img_l = np.array(Image.open(self.imgs_paths_l[idx]).convert("RGB"))
        img_l = np.max(img_l, axis=2)
        proj = np.array(Image.open(self.imgs_paths_proj[idx]).convert("RGB"))
        proj = np.max(proj, axis=2)
        img = np.dstack((img_l, proj, np.zeros_like(img_l)))
         
        return img

    def get_np_mask(self, idx):
        seg = np.array(Image.open(self.segs_paths[idx]).convert("L"))
        seg = seg.clip(max=1)
        return seg



    


def test_double_scan_v1():
    image_size = (1024,1024)
    train_transforms = A.Compose([
            A.Resize(image_size[0],image_size[1]),
            #A.CLAHE (clip_limit=(3.0,3.0), tile_grid_size=(8, 8), always_apply=True),
            #A.Blur(blur_limit=(5,5), always_apply=True),
            #A.RandomBrightnessContrast (brightness_limit=0.0, contrast_limit=(0.25, 0.25), always_apply=True),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.15,
                rotate_limit=10,
                p=0.7, border_mode=0),
            #ToTensorV2()
            ])

    ds = DoubleScanV1(train_transforms)
    img,seg = ds.get_np_img(0), ds.get_np_mask(0)
    #print(torch.max(seg))
    print(img.shape)
    print(seg.shape)
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(seg)
    plt.show()


    print("loader testing")
    train_loader, val_loader = get_double_scan_v1_loader(8, train_transforms)
    batch = next(iter(train_loader))
    train_batch, seg_batch = batch
    print(train_batch.shape)
    print(torch.min(train_batch))
    print(torch.max(train_batch))
    batch = next(iter(val_loader))
    train_batch, seg_batch = batch
    print(train_batch.shape)
    print(torch.min(train_batch))
    print(torch.max(train_batch))



    
if __name__ == '__main__':
    test_double_scan_v1()

    
