import sys
sys.path.append("utils")
sys.path.append("models")
from file_io import *
from train_utils import *


import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn
from random_word import RandomWords
import pandas as pd

from dataloaders import *
#from Unet2D import Unet2D

import torch.optim as optim
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
import time
import os
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
import torchgeometry
from logger_utils import *

rz = torchvision.transforms.Resize((512,512))

def row_wise_mean_index(img):
    try:
        img = np.average(img, axis=2)
    except:
        print("grayscale image")
    ret_img = np.zeros(np.shape(img))
    n_cols = np.shape(img)[1]
    col_inds = np.array(list(range(n_cols)))
    for r in range(len(img)):
        row = img[r]
        max_val = np.max(row)
        if max_val > 1:
            row_sum = np.sum(row)
            weighted_sum = np.argmax(row)
            ind = int(weighted_sum)
            ret_img[r,ind] = 255
    return ret_img



def test(model, test_dl, other_logdir):

    
    print(other_logdir)
    
    with torch.no_grad():
        for i,X_batch in enumerate(test_dl):
            X_batch = X_batch.cuda()
            
            outputs = model(X_batch)

            """
            X_batch_rz = rz(X_batch)
            print(X_batch_rz.shape)
            outputs_rz = rz(outputs)
            print(outputs_rz.shape)
            catted = torch.cat((X_batch_rz,outputs_rz), 0)
            print(catted.shape)
            """

            """
            grid = torchvision.utils.make_grid(catted)
            savedir = os.path.join(other_logdir, "test")
            os.makedirs(savedir, exist_ok=True)
            savepath = os.path.join(savedir, f"img{i}.png")
            torchvision.utils.save_image(grid, fp=savepath)
            """
            a = outputs.cpu().numpy()
            x = X_batch.cpu().numpy()
            
            for i in range(a.shape[0]):
                fig,ax = plt.subplots(1,3)
                fig.set_size_inches(18.5, 10.5)
                img = a[i][1]
                img2 = row_wise_mean_index(img)

                img_x = x[i,:,:,:]
                img_x = np.transpose(img_x, (1,2,0))
                
                ax[0].imshow(img_x)
                ax[0].axis('off')
                ax[1].imshow(img, cmap="hot")
                ax[1].axis('off')
                ax[2].imshow(img2, cmap="hot")
                ax[2].axis('off')
                plt.show()
            print(a.shape)
            save_batch_as_image(X_batch, torch.zeros((X_batch.shape[0], X_batch.shape[2],X_batch.shape[3]), dtype=torch.torch.int64), outputs, i, "Test", other_logdir)


            






def init_test(cfg, model_folder, hparam_search_dir=""):
    print("Model folder")
    print(model_folder)

    bs = 4
    val_transforms = cfg["val_transforms"]
    model_file = cfg["model"]
    dataset = cfg["dataset"]
    channel_ratio = cfg["channel_ratio"]
    cross_entr_weights = cfg["cross_entr_weights"]

    other_logdir = os.path.join("logdir", "other", dataset, hparam_search_dir, model_file)
    other_logdir = os.path.join(other_logdir, model_folder)
    model_state_dict_path = os.path.join(other_logdir, "state_dict.pth")
    print("Save model path")
    print(model_state_dict_path)

    model_path = os.path.join("models",dataset)
    model_import = import_model_from_path(model_file, model_path)


    unet = model_import.Unet2D(3,2, channel_ratio)
    unet.load_state_dict(torch.load(model_state_dict_path))
    unet.cuda()

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cross_entr_weights).cuda())

    test_loader = get_realscan_v1_loader(4, val_transforms)

    test(unet, test_loader, other_logdir)


if __name__ == '__main__':
    args = add_config_parser_with_model_folder()
    cfg = get_dict(args, print_config=True)
    model_folder = args.model_folder
    init_test(cfg, model_folder)

