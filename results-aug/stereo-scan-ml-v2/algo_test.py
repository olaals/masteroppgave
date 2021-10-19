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
        class_dice_score = np.sum(intersect)*2 / (sum_binary_gt+sum_binary_seg)
        return class_dice_score

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



def stack_ground_truth_pred(gt, pred, scan_img, show_imgs=False):
    scan_img = scan_img.astype(np.uint16)
    scan_img = scan_img *4
    scan_img[scan_img>254] = 255
    scan_img = scan_img.astype(np.uint8)
    binary_gt = np.where(gt>0, 1, 0)
    binary_pred = np.where(pred>0, 1, 0)
    stacked_gt_pred = np.dstack((np.zeros_like(binary_gt), binary_pred*255, binary_gt*255))
    stacked_gt_pred = stacked_gt_pred.astype(np.uint8)
    stacked_gt_pred  = shift_hue(stacked_gt_pred, 120)

    if show_imgs:
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(stacked_gt_pred)
        ax[1].imshow(scan_img)
        ax[2].imshow(proj_img)
        plt.show()
    return stacked_gt_pred




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


def save_images(dice, left_scan, right_scan, prediction, gt):
    pass
    

def test_dataset(dataset_path, test_path):
    SHOW_IMAGES = False
    MIN_OUTLIER_ERROR = 5
    result_dir = os.path.join("results", dataset_path)
    try:
        os.rmdir(result_dir)
        print("Removed directory", result_dir)
    except:
        print("No directory named", result_dir)



    gt_paths = get_named_image_from_dirs(dataset_path, "gt_scan.png")

    left_scans_paths = get_named_image_from_dirs(dataset_path, "img_l.png")
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

        gt_img = cv2.imread(gt_path, 0)
        gt_img = np.where(gt_img>10, gt_img, 0)
        subpix = secdeg_momentum_subpix(gt_img)
        pred_img = cv2.imread(pred_path, 0)
        left_scan = cv2.cvtColor(cv2.imread(left_scan_path), cv2.COLOR_BGR2RGB)

        pred_img = np.where(pred_img>15, pred_img, 0)


        
        binary_gt = np.where(gt_img>5, 1, 0)
        binary_pred = np.where(pred_img>0, 1, 0)

        #overlap_pred_gt(binary_gt, binary_pred)

        


        filtered=pred_img




        dice_score = get_dice_scores(binary_pred, binary_gt, 2)
        #print("dice:", dice_score)
        stacked = stack_ground_truth_pred(gt_img, pred_img, left_scan)

        dice_score_list.append(dice_score)
        #print("dice score", dice_score)

        combined_rgb = np.dstack((binary_pred*255, np.zeros_like(binary_pred), binary_gt*255))

        subpix_pred = secdeg_momentum_subpix(filtered, 0.3)


        subpix_score, outlier_fraction = get_subpix_score(subpix_pred, subpix, MIN_OUTLIER_ERROR)
        subpix_score_list.append(subpix_score)
        outlier_frac_list.append(outlier_fraction)

        #outlier_fraction = get_outlier_fraction(subpix_pred, subpix, 10)
        result_scan_dir = os.path.join(result_dir, "scan"+str(dice_score)+"-"+str(idx))
        os.makedirs(result_scan_dir, exist_ok=True)
        cv2_imwrite(os.path.join(result_scan_dir, "gt_scan.png"), gt_img)
        cv2_imwrite(os.path.join(result_scan_dir, "pred.png"), pred_img)
        cv2_imwrite(os.path.join(result_scan_dir, "left_scan.png"), left_scan)
        cv2_imwrite(os.path.join(result_scan_dir, "stacked.png"), stacked)






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

    outlier_frac_list.sort()
    dice_score_list.sort()
    subpix_score_list.sort()

    print("Avg dice score", avg_dice_score)
    print("avg subpix score", avg_subpix_score)
    print("avg outlier frac", format(avg_outlier_frac*100.0, "02f"), "%")
    return dice_score, subpix_score_list, dice_score_list, outlier_frac_list




    



if __name__ == '__main__':
    specular_path = os.path.join("test-sets", "test-specular")
    pbr_path = os.path.join("test-sets", "test-pbr")
    blurry_path = os.path.join("test-sets", "test-blurry")
    print("Specular")
    dice_score, subpix_score_list_spec, dice_scores_spec, outliers_spec= test_dataset(specular_path, "test-spec")
    print("PBR")
    dice_score, subpix_score_list_pbr, dice_scores_pbr, outliers_pbr = test_dataset(pbr_path, "test-pbr")
    print("Blurry")
    dice_score, subpix_score_list_blurry, dice_scores_blur, outliers_blur  = test_dataset(blurry_path, "test-blurry")



    fig,ax =plt.subplots(1,3)
    ax[0].plot(dice_scores_spec, label='Specular')
    ax[0].plot(dice_scores_pbr, label='PBR')
    ax[0].plot(dice_scores_blur, label='Blurry')
    ax[0].set_xlabel("Sorted images by dice score")
    ax[0].set_ylabel("Dice score")
    ax[0].legend()

    ax[1].plot(subpix_score_list_spec, label='Specular')
    ax[1].plot(subpix_score_list_pbr, label='PBR')
    ax[1].plot(subpix_score_list_blurry, label='Blurry')
    ax[1].set_xlabel("Sorted images by error")
    ax[1].set_ylabel("Mean image subpixel error")
    ax[1].legend()

    ax[2].plot(np.array(outliers_spec)*100, label='Specular')
    ax[2].plot(np.array(outliers_pbr)*100, label='PBR')
    ax[2].plot(np.array(outliers_blur)*100, label='Blurry')
    ax[2].set_xlabel("Sorted images by outlier fraction")
    ax[2].set_ylabel("Outlier fraction %")
    ax[2].legend()
    plt.show()







