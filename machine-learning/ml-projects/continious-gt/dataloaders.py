import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import glob
from skimage import io


import matplotlib.pyplot as plt



class CustomDataset(Dataset):
    def __init__(self, train_dir, gt_dir, transform=transforms.ToTensor()):
        super().__init__()
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.train_paths = glob.glob(train_dir + "/*.png")
        self.gt_paths = glob.glob(gt_dir + "/*.png")
        self.train_paths.sort()
        self.gt_paths.sort()
        self.length = min(len(self.train_paths), len(self.gt_paths))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        train_image = io.imread(self.train_paths[index])
        gt_image = io.imread(self.gt_paths[index])
        gt_image = np.argmax(gt_image, axis=1)
        gt_image = np.expand_dims(gt_image, 1)
        gt_image = gt_image.astype(np.float32)

        if self.transform:
            train_image = self.transform(train_image)
            gt_image = self.transform(gt_image)
            gt_image = torch.flatten(gt_image)
        #print(type(train_image))

        return (train_image, gt_image)

def get_dataloaders(train_dir, gt_dir, batch_size, validation_fraction=0.2):

    dataset = CustomDataset(train_dir, gt_dir)
    num_val_images = int(validation_fraction*len(dataset))
    num_train_images = len(dataset) - num_val_images

    train_set, val_set = torch.utils.data.random_split(dataset, [num_train_images, num_val_images])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


if __name__ == '__main__':
    import argparse
    import yaml
    from utils import get_tensor_as_image_grayscale, get_tensor_as_image
    import matplotlib.pyplot as plt

    # Load and print configuration file
    description = "Train script for pytorch"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config', help="Path to configuration yaml file")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print("Configuration parameters")
    for key, value in config.items():
        print(f'{key}: {value}')
    print("")

    train_dir_path = config["train_dir"]
    gt_dir_path = config["gt_dir"]

    dataset = CustomDataset(train_dir_path, gt_dir_path, transform = None)
    train0, val0 = dataset[0]
    plt.imshow(train0)
    plt.show()

    print(dataset)




