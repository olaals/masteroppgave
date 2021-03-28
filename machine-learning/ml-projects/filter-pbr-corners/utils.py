import numpy as np
import torch
import argparse
import yaml
import glob
from skimage import io
import matplotlib.pyplot as plt

def add_config_parser():
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
    return config

def scale_image(img):
    img = np.copy(img)
    from_max = np.max(img)
    from_min = np.min(img)
    to_max = 1.0
    to_min = 0.0
    img = to_min + (img-from_min)*(to_max-to_min)/(from_max-from_min)
    return img

def plot_2d_list_of_images(list_of_lists_of_imgs, save_path=None):
    plt.rcParams["figure.figsize"] = (20,20)
    l_imgs = list_of_lists_of_imgs
    num_vert_imgs = len(l_imgs)
    num_hori_imgs = len(l_imgs[0])
    fig, axs = plt.subplots(num_vert_imgs, num_hori_imgs)
    fig.subplots_adjust(hspace=0, wspace=0)
    img_iter = 0
    for r in range(num_vert_imgs):
        print(r)
        for c in range(num_hori_imgs):
            img_to_plot = np.copy(l_imgs[r][c])
            axs[r,c].axis('off')
            if len(np.shape(img_to_plot)) == 2:
                axs[r,c].imshow(l_imgs[r][c], cmap='gray')
            elif np.shape(img_to_plot)[2] == 3:
                axs[r,c].imshow(l_imgs[r][c])
            else:
                print("Something weird with the image")
                assert(False)
            img_iter += 1
    if save_path:
        fig.savefig(save_path + ".png", format='png')

def plot_1d_list_of_images(list_of_images, save_path=None):
    plt.rcParams["figure.figsize"] = (20,20)
    l_imgs = list_of_images
    num_imgs = len(l_imgs)
    fig, axs = plt.subplots(num_imgs)
    fig.subplots_adjust(hspace=0, wspace=0)
    img_iter = 0
    for i in range(num_imgs):
        img_to_plot = np.copy(l_imgs[i])
        axs[i].axis('off')
        if len(np.shape(img_to_plot)) == 2:
            axs[i].imshow(l_imgs[i], cmap='gray')
        elif np.shape(img_to_plot)[2] == 3:
            axs[i].imshow(l_imgs[i])
        else:
            print("Something weird with the image")
            assert(False)
        img_iter += 1
    if save_path:
        fig.savefig(save_path + ".png", format='png')
    else:
        plt.show()



    



def row_wise_max(img, threshold, img_type_float=True):
    img = np.copy(img)
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
        if max_val > threshold:
            ind = np.argmax(row)
            
            if img_type_float:
                ret_img[r,ind]=1.0
            else:
                ret_img[r,ind]=255
    return ret_img



def get_tensor_as_image(tensor, img_width):
    tensor = tensor.clone()
    tensor = tensor[0]
    np_img = tensor.cpu().data.numpy()
    np_img = np.copy(np_img)
    np_img = np.reshape(np_img, (3,img_width,-1))
    np_img = np.moveaxis(np_img,0,2)
    np_img = scale_image(np_img)
    return np_img

def get_tensor_as_image_grayscale(tensor, img_width):
    tensor = tensor.clone()
    tensor = tensor[0]
    np_img = tensor.cpu().data.numpy()
    np_img = np.copy(np_img)
    np_img = np.reshape(np_img, (img_width,-1))
    np_img = scale_image(np_img)
    return np_img

def image_stats(img):
    data_type = type(img[0][0])
    img_width = np.shape(img)[0]
    img_height = np.shape(img)[1]
    max_pix = np.max(img)
    min_pix = np.min(img)
    img_mean = np.mean(img)
    img_std = np.std(img)
    print(f'Type: {data_type}, Width: {img_width}, Height: {img_height}, Max: {max_pix}, Min: {min_pix}, Mean: {img_mean}, Std: {img_std}')

def tensor_stats(tensor):
    shape = tensor.shape
    tensor_max = torch.max(tensor)
    tensor_min = torch.min(tensor)
    tensor_mean = torch.mean(tensor)
    tensor_std = torch.std(tensor)
    print(f"Tensor stats: Max: {tensor_max}, Min: {tensor_min}, Mean: {tensor_mean}, Std: {tensor_std}")

def overlap_grayscale_images(img1, img2):
    img1 = np.copy(img1)
    img2 = np.copy(img2)
    assert(np.shape(img1) == np.shape(img2))
    black = np.zeros(np.shape(img1))
    img1 = scale_image(img1)
    img2 = scale_image(img2)

    return np.dstack((img1,img2,black))
    

def calculate_mean_and_std_dev(dataset_path, print_image_info=False, read_grey=False):
    image_path_list = glob.glob(train_dir_path + "/*.png")
    cumulative_sums = [0,0,0]
    cumulative_num_px = 0
    for ind, path in enumerate(image_path_list):
        image = io.imread(path, as_gray=read_grey)
        try: 
            image_channels = np.shape(image)[2]
        except:
            image_channels = 1
        
        img_height = np.shape(image)[0]
        img_width = np.shape(image)[1]

        if image_channels == 1:
            img_mean = np.mean(image)
            img_std = np.std(image)
            max_pix = np.max(image)
            min_pix = np.min(image)
            cumulative_sums[0] += np.sum(image)
            if print_image_info:
                print(f'Image number: {ind}, Mean: {img_mean}, Std: {img_std}, Max: {max_pix}, Min: {min_pix}')
        if image_channels == 3:
            for chan in range(3):
                img_chan = image[:,:,chan]
                cumulative_sums[chan] += np.sum(img_chan)
                chan_mean = np.mean(img_chan)
                chan_std = np.std(img_chan)
                max_pix = np.max(img_chan)
                min_pix = np.min(img_chan)
                if print_image_info:
                    print(f'Image number: {ind}, Channel: {chan}, Mean: {chan_mean}, Std: {chan_std}, Max: {max_pix}, Min: {min_pix}')


        cumulative_num_px += img_height*img_width*image_channels
        if print_image_info:
            print(f'Width: {img_width}, Height: {img_height}, Channels: {image_channels}')

    dataset_mean = np.array(cumulative_sums)/cumulative_num_px
    


    
    cumulative_sum = np.zeros(3)
    for ind, path in enumerate(image_path_list):
        image = io.imread(path, as_gray=read_grey)
        try: 
            image_channels = np.shape(image)[2]
        except:
            image_channels = 1
        if image_channels == 1:
            cumulative_sum[0] += np.sum(image)
        if image_channels == 3:
            for chan in range(image_channels):
                img_chan = image[:,:,chan]
                cumulative_sum[chan] += np.sum((img_chan - dataset_mean[chan])**2)


    print(cumulative_sum)

    dataset_std = np.sqrt(cumulative_sum/cumulative_num_px)
    





    print('')
    print(f'Dataset mean: {dataset_mean}')
    print(f'Dataset std: {dataset_std}')
        







if __name__ == '__main__':
    config =  add_config_parser()
    train_dir_path = config["train_dir"]
    gt_path = config["gt_dir"]

    #calculate_mean_and_std_dev(train_dir_path)
    print(gt_path)
    calculate_mean_and_std_dev(gt_path, True, True)





