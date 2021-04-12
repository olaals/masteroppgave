import time
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
import yaml

# Same directory imports
sys.path.append('models')
from dataloaders import get_dataloaders
from model import CustomModel
from utils import *

# Run on GPU
assert(torch.cuda.is_available())
device = torch.device('cuda')
print(f'Training on {torch.cuda.get_device_name(0)}\n')

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

batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
input_channels = config["input_channels"]
input_width, input_height = config["input_width"], config["input_height"]
num_epochs = config["epochs"]
val_frac = config["validation_fraction"]
train_dir_path = config["train_dir"]
gt_dir_path = config["gt_dir"]


# Get dataloaders
train_loader, val_loader = get_dataloaders(train_dir_path, gt_dir_path, batch_size, val_frac)


# Import model
model = CustomModel()
model.cuda()

# Set criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

since = time.time()


# Train
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    running_loss = 0.0
    counter = 0
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for X_batch, Y_batch in tk0:
        X_batch = X_batch.to(device='cuda')
        Y_batch = Y_batch.to(device='cuda')
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * X_batch.size(0)
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * X_batch.size(0))))
    epoch_loss = running_loss / len(train_loader)*train_loader.batch_size
    print('Training Loss: {:.4f}'.format(epoch_loss))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
torch.save(model.state_dict(), "model.bin")
