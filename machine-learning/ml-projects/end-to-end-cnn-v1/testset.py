import sys
sys.path.append("utils")
sys.path.append("models")
from file_io import *
from train_utils import *
import cv2


import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time
from test import init_test

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn
from random_word import RandomWords
import pandas as pd

from dataloaders import get_double_scan_v1_loader
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
np.random.seed(int(time.time()))



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

def show_images(outputs,X_batch,Y_batch, other_logdir, idx):
    os.makedirs(other_logdir, exist_ok=True)
    a = outputs.cpu().numpy()
    x = X_batch.cpu().numpy()
    y = Y_batch.cpu().numpy()
    
    for i in range(a.shape[0]):
        #fig,ax = plt.subplots(1,3)
        #fig.set_size_inches(18.5, 10.5)
        img = a[i][1]
        gt = y[i]
        img2 = row_wise_mean_index(img)

        img_x = x[i,:,:,:]
        img_x = np.transpose(img_x, (1,2,0))
        
        img = (img*254/np.max(img)).astype(np.uint8)
        gt = (gt*254/np.max(gt)).astype(np.uint8)
        cv2.imwrite(os.path.join(other_logdir, "img"+format(idx+i, '02d')+".png"), img)
        #cv2.imwrite(os.path.join(other_logdir, "gt"+format(idx+i, '02d')+".png"), gt)
        """
        ax[0].imshow(img_x)
        ax[0].axis('off')
        ax[1].imshow(img, cmap="hot")
        ax[1].axis('off')
        ax[2].imshow(img2, cmap="hot")
        ax[2].axis('off')
        plt.show()
        """


def check_accuracy(valid_dl, model, loss_fn, classes, other_logdir):
    print("Other logdir", other_logdir)
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    running_class_dice = np.zeros(classes)
    save_batch = True
    batch_idx = 0
    with torch.no_grad():
        for X_batch, Y_batch in valid_dl:
            print(X_batch.shape)
            X_batch = X_batch.cuda()
            Y_batch = Y_batch.cuda()
            cur_batch_sz = X_batch.size(0)
            outputs = model(X_batch)
            show_images(outputs, X_batch, Y_batch, other_logdir, batch_idx)
            loss = loss_fn(outputs, Y_batch.long())
            dice_score, dice_class_scores = mean_dice_score(outputs, Y_batch, classes)
            running_loss += loss * cur_batch_sz
            running_dice += dice_score * cur_batch_sz
            running_class_dice += dice_class_scores * cur_batch_sz
            batch_idx += X_batch.size(0)

    average_loss = running_loss / len(valid_dl.dataset)
    average_acc = running_acc / len(valid_dl.dataset)
    average_dice_sc = running_dice / len(valid_dl.dataset)
    average_dice_class_sc = running_class_dice / len(valid_dl.dataset)

    print('{} Loss: {:.4f} Dice: {}'.format("Test", average_loss, average_dice_sc))
    return average_dice_sc, average_dice_class_sc



def test(model, test_dl, other_logdir):

    
    print(other_logdir)
    
    with torch.no_grad():
        for i,(X_batch, Y_batch) in enumerate(test_dl):
            X_batch = X_batch.cuda()
            
            outputs = model(X_batch)

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

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cross_entr_weights).cuda())

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



    test_blurry, test_specular, test_pbr = get_test_blurry_loader(4)

    #test(unet, test_loader, other_logdir)
    print("Test pbr")
    check_accuracy(test_pbr, unet, loss_fn, 1, other_logdir+"/test/test-pbr")
    print("Test specular")
    check_accuracy(test_specular, unet, loss_fn, 1, other_logdir+"/test/test-spec")
    print("Test blurry")
    check_accuracy(test_blurry, unet, loss_fn, 1, other_logdir+"/test/test-blurry")


if __name__ == '__main__':
    args = add_config_parser_with_model_folder()
    cfg = get_dict(args, print_config=True)
    model_folder = args.model_folder
    init_test(cfg, model_folder)

