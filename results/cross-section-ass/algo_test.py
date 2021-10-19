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
            return 0.9
        class_dice_score = np.sum(intersect)*2 / (sum_binary_gt+sum_binary_seg)
        dice_scores.append(class_dice_score)
        return class_dice_score

"""
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
"""


"""
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

"""


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


    

def test_dataset(dataset_path):
    SHOW_IMAGES = False

    nonzero_avg_paths = get_named_image_from_dirs(dataset_path, "nonzero_avg.png")
    left_imgs_paths = get_named_image_from_dirs(dataset_path, "img_l.png")
    proj_imgs_paths = get_named_image_from_dirs(dataset_path, "proj.png")
    gt_paths = get_named_image_from_dirs(dataset_path, "gt_scan.png")
    subpix_paths = get_named_image_from_dirs(dataset_path, "subpix.npy")

    result_dir = os.path.join("results", dataset_path)
    try:
        os.rmdir(result_dir)
        print("Removed directory", result_dir)
    except:
        print("No directory named", result_dir)

    dice_score_list = []
    subpix_score_list =  []
    outlier_frac_list = []

    for idx in range(len(nonzero_avg_paths)):
        nonzero_avg_path = nonzero_avg_paths[idx]
        gt_path = gt_paths[idx]
        img_l_path = left_imgs_paths[idx]
        subpix_path = subpix_paths[idx]
        proj_path = proj_imgs_paths[idx]

        subpix = np.load(subpix_path)
        nonzero_avg_img = cv2.imread(nonzero_avg_path,0)
        gt_img = cv2.imread(gt_path, 0)
        img_l = cv2.imread(img_l_path, 0)
        proj = cv2.imread(proj_path, 0)

        right_line_img = right_line_mask(nonzero_avg_img)

        filtered = np.where(right_line_img, nonzero_avg_img, 0)

        binary_pred = np.where(right_line_img>0, 1, 0)
        
        binary_gt = np.where(gt_img>20, 1, 0)

        stacked_pred_gt = np.dstack((binary_pred*255, binary_gt*255, np.zeros_like(binary_gt))).astype(np.uint8)
        stacked_left_proj = np.dstack((proj, img_l, np.zeros_like(proj)))

        dice_score = get_dice_scores(binary_pred, binary_gt, 2)
        dice_score_list.append(dice_score)
        #print("dice score", dice_score)

        combined_rgb = np.dstack((binary_pred*255, np.zeros_like(binary_pred), binary_gt*255))

        subpix_pred = secdeg_momentum_subpix(filtered, 0.3)


        subpix_score, outlier_fraction = get_subpix_score(subpix_pred, subpix, 5)
        subpix_score_list.append(subpix_score)
        outlier_frac_list.append(outlier_fraction)

        result_scan_dir = os.path.join(result_dir, "scan"+str(dice_score)+"-"+str(idx))
        os.makedirs(result_scan_dir, exist_ok=True)
        cv2_imwrite(os.path.join(result_scan_dir, "stacked_pred_gt.png"), stacked_pred_gt)
        cv2_imwrite(os.path.join(result_scan_dir, "stacked.png"), stacked_left_proj)
        nonzero_avg_img = nonzero_avg_img.astype(np.uint8)
        cv2.imwrite(os.path.join(result_scan_dir, "nonzero_avg.png"), nonzero_avg_img)
        right_line_img = (right_line_img*255).astype(np.uint8)
        cv2.imwrite(os.path.join(result_scan_dir, "right_line.png"), right_line_img)

        #outlier_fraction = get_outlier_fraction(subpix_pred, subpix, 10)





        if SHOW_IMAGES:
            print("subpix", subpix.shape)
            print("subpix_pred", subpix_pred.shape)
            plt.imshow(combined_rgb)
            plt.show()
            plt.imshow(filtered)
            plt.show()

    subpix_score_list = np.sort(np.array(subpix_score_list))
    subpix_score_list.sort()
    outlier_frac_list = np.sort(np.array(outlier_frac_list))*100.0

    dice_scores_np = np.sort(np.array(dice_score_list))
    dice_scores_np.sort()
    avg_dice_score = np.average(dice_scores_np)

    avg_subpix_score = np.mean(subpix_score_list)
    avg_outlier_frac = np.mean(outlier_frac_list)

    print("Avg dice score", avg_dice_score)
    print("avg subpix score", avg_subpix_score)
    print("avg outlier frac", avg_outlier_frac)
    return dice_score, subpix_score_list, dice_scores_np, outlier_frac_list




    



if __name__ == '__main__':

    specular_path = os.path.join("test-sets", "test-specular")
    pbr_path = os.path.join("test-sets", "test-pbr")
    blurry_path = os.path.join("test-sets", "test-blurry")
    print("Specular")
    dice_score, subpix_score_list_spec, dice_scores_spec, outliers_spec= test_dataset(specular_path)
    print("PBR")
    dice_score, subpix_score_list_pbr, dice_scores_pbr, outliers_pbr = test_dataset(pbr_path)
    print("Blurry")
    dice_score, subpix_score_list_blurry, dice_scores_blur, outliers_blur  = test_dataset(blurry_path)

    dice_scores_spec.sort()
    dice_scores_blur.sort()
    dice_scores_pbr.sort()
    print("here")
    print(dice_scores_blur.shape)



    fig,ax =plt.subplots(1,3)
    ax[0].plot(dice_scores_spec, label='Specular')
    ax[0].plot(dice_scores_pbr, label='PBR')
    ax[0].plot(dice_scores_blur, label='Blurry')
    ax[0].set_xlabel("Sorted images by dice score")
    ax[0].set_ylabel("Dice score")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(subpix_score_list_spec, label='Specular')
    ax[1].plot(subpix_score_list_pbr, label='PBR')
    ax[1].plot(subpix_score_list_blurry, label='Blurry')
    ax[1].set_xlabel("Sorted images by error")
    ax[1].set_ylabel("Mean image subpixel error")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(outliers_spec), label='Specular')
    ax[2].plot(np.array(outliers_pbr), label='PBR')
    ax[2].plot(np.array(outliers_blur), label='Blurry')
    ax[2].set_xlabel("Sorted images by outlier fraction")
    ax[2].set_ylabel("Outlier fraction %")
    ax[2].legend()
    ax[2].grid(True)
    plt.show()


    """
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

    """

