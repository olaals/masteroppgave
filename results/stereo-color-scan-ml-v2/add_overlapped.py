import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from oa_dev import *
from oa_filter import *
from oa_ls import *



def get_scan_paths(dataset_dir):
    scan_dirs = [os.path.join(dataset_dir, scan_dir) for scan_dir in os.listdir(dataset_dir)]
    scan_dirs.sort()
    return scan_dirs


def check_rgb_overlap_corr(scan, corr1):
    corr = corr1.copy()
    corr = cv2.blur(corr, (3,3))
    thresh = 10
    corr_r = corr[:,:,0]>thresh
    corr_g = corr[:,:,1]>thresh
    corr_b = corr[:,:,2]>thresh
    r = np.where(corr_r, scan[:,:,0], 0)
    g = np.where(corr_g, scan[:,:,1], 0)
    b = np.where(corr_b, scan[:,:,2], 0)


    combined = np.dstack((r,g,b))

    return combined


def main(input_dir):
    scan_dirs = get_scan_paths(input_dir)

    for scan_dir in scan_dirs:
        print(scan_dir)
        corr_path = os.path.join(scan_dir, "corr.png")
        proj_path = os.path.join(scan_dir, "proj.png")
        img_l_path = os.path.join(scan_dir, "img_l.png")
        gt_mask_path = os.path.join(scan_dir, "gt_proj_mask.png")

        corr = cv2.cvtColor(cv2.imread(corr_path), cv2.COLOR_BGR2RGB)
        proj = cv2.cvtColor(cv2.imread(proj_path), cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(cv2.imread(img_l_path), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(gt_mask_path, 0)

        combined = check_rgb_overlap_corr(img_l, corr)
        combined_proj = check_rgb_overlap_corr(proj, corr)

        print(np.mean(combined[combined>0]))


        comb_stereo = np.zeros_like(combined)

        for c in range(3):
            comb_stereo = comb_stereo.astype(np.float32)
            comb_stereo[:,:,c] = np.power(combined[:,:,c]/255.0, 1)*np.power(combined_proj[:,:,c]/255.0,1)
            print(np.max(comb_stereo))
            #comb_stereo = np.clip(comb_stereo, 0, 255)
            #comb_stereo = comb_stereo.astype(np.uint8)

            print(np.max(comb_stereo))

        comb_stereo = comb_stereo/np.max(comb_stereo)*254
        a = 0.33
        b = 0.33
        c = 0.33
        comb_stereo = a*comb_stereo[:,:,0]+b*comb_stereo[:,:,1]+c*comb_stereo[:,:,2]
        comb_stereo = comb_stereo.astype(np.uint8)
        
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
        #comb_stereo = clahe.apply(comb_stereo)
        comb_stereo = np.where(comb_stereo>5, comb_stereo, 0)



        #stack = np.dstack(((comb_stereo>10)*255, (gt_mask>0)*255, np.zeros_like(gt_mask)))


        """
        fig,ax = plt.subplots(1,5)
        ax[0].imshow(corr)
        ax[1].imshow(combined)
        ax[2].imshow(combined_proj)
        ax[3].imshow(comb_stereo)

        plt.show()
        """

        

        #cv2.imwrite(os.path.join(scan_dir, "stereo-filt.png"),comb_stereo)
        cv2.imwrite(os.path.join(scan_dir, "filt_rgb.png"),combined)
        cv2.imwrite(os.path.join(scan_dir, "filt_proj_rgb.png"),combined_proj)

        



if __name__ == '__main__':
    scan_dirs = "double-stereo-scan-v1"
    pbr_dir = os.path.join("test-sets", "test-color-pbr")
    spec_dir = os.path.join("test-sets", "test-color-specular")
    blur_dir = os.path.join("test-sets", "test-color-blurry")
    dirs = [scan_dirs, pbr_dir, spec_dir, blur_dir]
    print(dirs)
    for scan_dir in dirs:
        main(scan_dir)
