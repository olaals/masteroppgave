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
    #print("get outlier fraction")
    pred_array[pred_array==0] = np.nan
    gt_array[gt_array==0] = np.nan



    both_nan = np.bitwise_and(np.isnan(pred_array), np.isnan(gt_array))



    pred_array=pred_array[~both_nan]
    gt_array=gt_array[~both_nan]

    diff = pred_array - gt_array
    diff = diff[~np.isnan(diff)]
    diff = np.abs(diff)
    outliers = diff>min_outlier_error

    outlier_frac = np.sum(outliers)*1.0/len(gt_array)
    #print(outlier_frac*100)
    return  outlier_frac


def shift_hue(img, shift):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = (hsv[:,:,0] + shift)%180
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb



def stack_ground_truth_pred(gt, pred, scan_img):
    scan_img = scan_img.astype(np.uint16)
    scan_img = scan_img *4
    scan_img[scan_img>254] = 255
    scan_img = scan_img.astype(np.uint8)
    binary_gt = np.where(gt>0, 1, 0)
    binary_pred = np.where(pred>0, 1, 0)
    fig,ax = plt.subplots(1,2)
    stacked_gt_pred = np.dstack((np.zeros_like(binary_gt), binary_pred*255, binary_gt*255))
    stacked_gt_pred = stacked_gt_pred.astype(np.uint8)
    stacked_gt_pred  = shift_hue(stacked_gt_pred, 120)

    ax[0].imshow(stacked_gt_pred)
    ax[1].imshow(scan_img)
    plt.show(sharex=True, sharey=True)



def plot_pred_gt(pred_array, gt_array):
    
    plt.plot(pred_array, label="Prediction")
    plt.plot(gt_array, label="Ground truth")
    plt.legend()
    plt.ylabel("Centre of scan line, column index")
    plt.xlabel("Row index")

    plt.show()

def overlap_pred_gt(binary_pred, binary_gt):
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(binary_pred)
    ax[1].imshow(binary_gt)
    ax[2].imshow(np.dstack((binary_gt*255, binary_pred*255, np.zeros_like(binary_gt))))
    plt.show()



def get_subpix_score(pred_array, gt_array, min_outlier_error):
    SHOW_IMAGES = False
    pred_array[pred_array==0] = np.nan
    gt_array[gt_array==0] = np.nan


    outlier_frac  =get_outlier_fraction(pred_array, gt_array, min_outlier_error)
    #print("Outlier frac:", outlier_frac)
    #plot_pred_gt(pred_array, gt_array)


    diff = np.abs(pred_array- gt_array)
    diff = diff[~np.isnan(diff)]
    diff = diff[diff<min_outlier_error]



    if SHOW_IMAGES:

        fig,ax = plt.subplots(1,2)
        ax[0].plot(pred_array)
        ax[0].plot(gt_array)

        ax[1].plot(diff)
        plt.show()

        print("mean subpix error", np.mean(diff))
    return np.mean(diff),outlier_frac


def get_hue_diff_img(proj, left):
    proj_hsv = cv2.cvtColor(proj, cv2.COLOR_RGB2HSV)
    left_hsv = cv2.cvtColor(left, cv2.COLOR_RGB2HSV)
    left_value_mask = left_hsv[:,:,2]>40
    proj_value_mask = proj_hsv[:,:,2]>40
    nonzero_mask = np.bitwise_and(left_value_mask, proj_value_mask)
    proj_hue = proj_hsv[:,:,0].astype(np.int16)
    print("max hue", np.max(proj_hue))
    left_hue = left_hsv[:,:,0].astype(np.int16)
    diff = proj_hue - left_hue
    print(diff.shape)
    diff = np.abs(diff)
    print("max diff", np.max(diff))
    print(diff.shape)
    diff_opp = 180-diff

    print(diff_opp.shape)

    hue_diff = np.minimum(diff, 180-diff)/90*255
    print("max hue_diff", np.max(hue_diff))
    hue_diff_filt = np.where(nonzero_mask, hue_diff, 0)
    invert_hue_diff = 180-hue_diff
    invert_hue_diff_nonzero = np.where(nonzero_mask, invert_hue_diff, 0)


    fig,ax = plt.subplots(1,4, sharex='all', sharey='all')
    ax[0].imshow(hue_diff_filt)
    ax[1].imshow(nonzero_mask*255)
    ax[2].imshow(proj)
    ax[3].imshow(left)
    plt.show()
    
    print(hue_diff.shape)


    
def check_rgb_overlap_corr(scan, corr):
    thresh = 30
    scan_r = scan[:,:,0]>thresh 
    corr_r = corr[:,:,0]>thresh
    scan_g = scan[:,:,1]>thresh
    corr_g = corr[:,:,1]>thresh
    scan_b = scan[:,:,2]>thresh
    corr_b = scan[:,:,2]>thresh
    r = np.where(corr_r, scan[:,:,0], 0)
    g = np.where(corr_g, scan[:,:,1], 0)
    b = np.where(corr_b, scan[:,:,2], 0)


    combined = np.dstack((r,g,b))




    return combined

def hue_diff_test(dataset_path, test_path):
    SHOW_IMAGES = False
    MIN_OUTLIER_ERROR = 5

    gt_paths = get_named_image_from_dirs(dataset_path, "gt_scan.png")
    gt_proj_mask_paths = get_named_image_from_dirs(dataset_path, "gt_proj_mask.png")

    left_scans_paths = get_named_image_from_dirs(dataset_path, "img_l.png")
    proj_scans_paths = get_named_image_from_dirs(dataset_path, "proj.png")
    corr_paths = get_named_image_from_dirs(dataset_path, "corr.png")
    testset_path = os.path.join("test", test_path)
    pred_paths = [os.path.join(testset_path, img) for img in os.listdir(testset_path)]
    pred_paths.sort()

    dice_score_list = []
    subpix_score_list =  []
    outlier_frac_list = []

    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        pred_path = pred_paths[idx]
        left_scan_path = left_scans_paths[idx]
        gt_proj_path = gt_proj_mask_paths[idx]
        proj_scan_path = proj_scans_paths[idx]
        corr_path = corr_paths[idx]


        gt_img = cv2.imread(gt_path, 0)
        gt_proj_mask = cv2.imread(gt_proj_path,0)
        subpix = secdeg_momentum_subpix(gt_img)
        pred_img = cv2.imread(pred_path, 0)
        left_scan = cv2.cvtColor(cv2.imread(left_scan_path), cv2.COLOR_BGR2RGB)
        proj_scan = cv2.cvtColor(cv2.imread(proj_scan_path), cv2.COLOR_BGR2RGB)
        corr = cv2.cvtColor(cv2.imread(corr_path), cv2.COLOR_BGR2RGB)

        #get_hue_diff_img(left_scan, corr)
        over_scan = check_rgb_overlap_corr(left_scan, corr)
        over_proj = check_rgb_overlap_corr(proj_scan, corr)
        #overlaped_and = np.bitwise_and(over_scan>0, over_proj>0)



        fig,ax = plt.subplots(1,4)
        ax[0].imshow(left_scan)
        ax[1].imshow(proj_scan)
        ax[2].imshow(over_scan)
        ax[3].imshow(over_proj)
        #ax[4].imshow(overlaped_and)
        plt.show()

    



if __name__ == '__main__':
    specular_path = os.path.join("test-sets", "test-color-specular")
    pbr_path = os.path.join("test-sets", "test-color-pbr")
    blurry_path = os.path.join("test-sets", "test-color-blurry")
    print("Specular")
    hue_diff_test(specular_path, "test-spec")
    #print("PBR")
    #hue_diff_test(pbr_path, "test-pbr")
    print("Blurry")
    hue_diff_test(blurry_path, "test-blurry")

