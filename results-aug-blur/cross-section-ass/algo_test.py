import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from oa_dev import *
from oa_filter import *
from oa_ls import *




def get_dice_scores(segmentation, ground_truth, classes):
    dice_scores = []
    for i in range(1,classes+1):
        binary_gt = (ground_truth == i).astype(np.uint8)
        binary_seg = (segmentation == i).astype(np.uint8)
        intersect = np.logical_and(binary_gt, binary_seg)
        sum_binary_gt = np.sum(binary_gt)
        sum_binary_seg = np.sum(binary_seg)
        if sum_binary_gt == 0:
            continue
        class_dice_score = np.sum(intersect)*2 / (sum_binary_gt+sum_binary_seg)
        dice_scores.append(class_dice_score)
    dice_scores = np.array(dice_scores)
    return dice_scores

def get_outlier_fraction(pred_array, gt_array, min_outlier_error):
    print("get outlier fraction")
    pred_array[pred_array==0] = np.nan
    gt_array[gt_array==0] = np.nan

    plt.plot(pred_array)
    plt.plot(gt_array)
    plt.show()

    both_nan = np.bitwise_and(np.isnan(pred_array), np.isnan(gt_array))
    pred_array=pred_array[~both_nan]
    gt_array=gt_array[~both_nan]
    plt.plot(pred_array)
    plt.plot(gt_array)
    plt.show()




def get_subpix_score(pred_array, gt_array, min_outlier_error):
    SHOW_IMAGES = False
    pred_array[pred_array==0] = np.nan
    gt_array[gt_array==0] = np.nan

    diff = np.abs(pred_array-gt_array)
    diff = diff[~np.isnan(diff)]
    diff = diff[diff<min_outlier_error]

    outlier_frac = np.mean(diff>min_outlier_error)

    if SHOW_IMAGES:

        fig,ax = plt.subplots(1,2)
        ax[0].plot(pred_array)
        ax[0].plot(gt_array)

        ax[1].plot(diff)
        plt.show()

        print("mean subpix error", np.mean(diff))
    return np.mean(diff),outlier_frac


    

def test_dataset(dataset_path):
    SHOW_IMAGES = False

    nonzero_avg_paths = get_named_image_from_dirs(dataset_path, "nonzero_avg.png")
    left_imgs_paths = get_named_image_from_dirs(dataset_path, "img_l.png")
    gt_paths = get_named_image_from_dirs(dataset_path, "gt_scan.png")
    subpix_paths = get_named_image_from_dirs(dataset_path, "subpix.npy")

    dice_score_list = []
    subpix_score_list =  []
    outlier_frac_list = []

    for idx in range(len(nonzero_avg_paths)):
        nonzero_avg_path = nonzero_avg_paths[idx]
        gt_path = gt_paths[idx]
        img_l_path = left_imgs_paths[idx]
        subpix_path = subpix_paths[idx]

        subpix = np.load(subpix_path)
        nonzero_avg_img = cv2.imread(nonzero_avg_path,0)
        gt_img = cv2.imread(gt_path, 0)
        img_l = cv2.imread(img_l_path, 0)

        right_line_img = right_line_mask(nonzero_avg_img)

        filtered = np.where(right_line_img, nonzero_avg_img, 0)

        binary_pred = np.where(right_line_img>0, 1, 0)
        
        binary_gt = np.where(gt_img>0, 1, 0)

        dice_score = get_dice_scores(binary_pred, binary_gt, 2)
        dice_score_list.append(dice_score)
        #print("dice score", dice_score)

        combined_rgb = np.dstack((binary_pred*255, np.zeros_like(binary_pred), binary_gt*255))

        subpix_pred = secdeg_momentum_subpix(filtered, 0.3)


        subpix_score, outlier_fraction = get_subpix_score(subpix_pred, subpix, 10)
        subpix_score_list.append(subpix_score)
        outlier_frac_list.append(outlier_fraction)

        #outlier_fraction = get_outlier_fraction(subpix_pred, subpix, 10)





        if SHOW_IMAGES:
            print("subpix", subpix.shape)
            print("subpix_pred", subpix_pred.shape)
            plt.imshow(combined_rgb)
            plt.show()
            plt.imshow(filtered)
            plt.show()

    subpix_score_list = np.sort(np.array(subpix_score_list))
    outlier_frac_list = np.array(outlier_frac_list)

    dice_scores_np = np.array(dice_score_list)
    avg_dice_score = np.average(dice_scores_np)

    avg_subpix_score = np.mean(subpix_score_list)
    avg_outlier_frac = np.mean(outlier_frac_list)

    print("Avg dice score", avg_dice_score)
    print("avg subpix score", avg_subpix_score)
    print("avg outlier frac", avg_outlier_frac)
    return dice_score, subpix_score_list




    



if __name__ == '__main__':
    specular_path = os.path.join("test-sets", "test-specular")
    pbr_path = os.path.join("test-sets", "test-pbr")
    blurry_path = os.path.join("test-sets", "test-blurry")
    #specular_path = os.path.join("test-sets", "test-specular")
    print("Specular")
    dice_score, subpix_score_list_spec, = test_dataset(specular_path)
    print("PBR")
    dice_score, subpix_score_list_pbr = test_dataset(pbr_path)
    print("Blurry")
    dice_score, subpix_score_list_blurry = test_dataset(blurry_path)


    plt.plot(subpix_score_list_spec, label="Specular")
    plt.plot(subpix_score_list_pbr, label="PBR")
    plt.plot(subpix_score_list_blurry, label="Blurry")
    plt.ylabel("Mean image subpixel error")
    plt.xlabel("Sorted images by error")
    plt.legend()
    plt.show()


