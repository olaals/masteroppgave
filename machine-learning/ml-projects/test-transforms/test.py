import torchvision
import torchvision.transforms as transforms
import glob
import torch
import sys
import os
from skimage import io
import cv2
import matplotlib.pyplot as plt
from importlib import import_module

sys.path.append('models/models')

from utils import *

config = add_config_parser()

batch_size = int(config["batch_size"])
learning_rate = float(config["learning_rate"])
input_channels = int(config["input_channels"])
input_width, input_height = int(config["input_width"]), int(config["input_height"])
model_save_dir = config["model_save_dir"]
model = config["model"]
test_dir_path = config["test_dir"]

model_save_path = os.path.join(model_save_dir, model) + ".pth"

# Import model
model_module = import_module(model)
model = model_module.CustomModel()

model.load_state_dict(torch.load(model_save_path))
model.eval()

test_images_list = glob.glob(test_dir_path + "/*.png")

test_transforms = transforms.ToTensor()

for i in range(len(test_images_list)):
    img = cv2.imread(test_images_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    tensor_img = test_transforms(img)
    tensor_img = tensor_img.unsqueeze(0)

    pred_tensor = model(tensor_img)

    pred_img_np = get_tensor_as_image_grayscale(pred_tensor, 500)

    row_max_img = row_wise_max(pred_img_np, 0.1)

    imgs = [img, pred_img_np, row_max_img]
    plot_1d_list_of_images(imgs)


