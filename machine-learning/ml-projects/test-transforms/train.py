import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import time
import os
from importlib import import_module

# Same directory imports
sys.path.append('models/models')
from dataloaders import get_dataloaders
#from model import CustomModel
from utils import *

# Run on GPU
assert(torch.cuda.is_available())
device = torch.device('cuda')
print(f'Training on {torch.cuda.get_device_name(0)}\n')

# Load and print configuration file
config = add_config_parser()

batch_size = int(config["batch_size"])
learning_rate = float(config["learning_rate"])
input_channels = int(config["input_channels"])
input_width, input_height = int(config["input_width"]), int(config["input_height"])
num_epochs = int(config["epochs"])
val_frac = float(config["validation_fraction"])
train_dir_path = config["train_dir"]
gt_dir_path = config["gt_dir"]
plots_dir = config["plots_dir"]
model_save_dir = config["model_save_dir"]
model = config["model"]
plots_path = os.path.join(plots_dir, model)
try:
    os.mkdir(plots_path)
except:
    print(plots_path + " already exits, overwriting")



model_save_path = os.path.join(model_save_dir, model) + ".pth"



# Get dataloaders
train_loader, val_loader = get_dataloaders(train_dir_path, gt_dir_path, batch_size, val_frac)


# Import model
model_module = import_module(model)
model = model_module.CustomModel()
model.cuda()

# Set criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train
since = time.time()
for epoch in range(num_epochs):
    model.train()
    show_img = True
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    running_loss = 0.0
    counter = 0
    tqdm_iter = tqdm(train_loader, total=int(len(train_loader)))
    for X_batch, Y_batch in tqdm_iter:
        X_batch = X_batch.to(device='cuda')
        Y_batch = Y_batch.to(device='cuda')
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * X_batch.size(0)
        counter += X_batch.size(0)
        tqdm_iter.set_postfix(loss=(running_loss / counter))
    epoch_loss = running_loss / counter
    print('Training Loss: {:.5f}'.format(epoch_loss))
    val_iter = 0
    model.eval()
    with torch.no_grad():
        val_iter += 1
        val_running_loss = 0
        save_imgs = []
        num_save_val_imgs = 3
        epoch_imgs = []
        val_counter = 0

        for X_batch, Y_batch in val_loader:
            val_counter += X_batch.size(0)
            X_batch = X_batch.to(device='cuda')
            Y_batch = Y_batch.to(device='cuda')
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            val_running_loss += loss.item() * X_batch.size(0)
            if epoch > -1 and num_save_val_imgs >= 0:
                num_save_val_imgs -= 1
                print("X_batch tensor")
                tensor_stats(X_batch)
                print("Y_batch tensor")
                tensor_stats(Y_batch)
                print("Outputs tensor")
                tensor_stats(outputs)
                show_img = False
                train_img = get_tensor_as_image(X_batch, input_width)
                #image_stats(train_img)
                output_img = get_tensor_as_image_grayscale(outputs, input_width)
                #image_stats(output_img)
                gt_img = get_tensor_as_image_grayscale(Y_batch, input_width)
                #image_stats(gt_img)
                output_row_max = row_wise_max(output_img, 0.1)
                overlap_img = overlap_grayscale_images(output_row_max, gt_img)

                save_imgs = [train_img, output_img, output_row_max, gt_img, overlap_img]
                epoch_imgs.append(save_imgs)

        plot_2d_list_of_images(epoch_imgs, plots_path + "/" + "val_epoch" + str(epoch))
        val_loss = val_running_loss / val_counter
        print('Validation Loss: {:.5f}'.format(val_loss))
        



time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
torch.save(model.state_dict(), model_save_path)
