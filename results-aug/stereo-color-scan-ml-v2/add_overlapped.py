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
    corr_b = scan[:,:,2]>thresh
    r = np.where(corr_r, scan[:,:,0], 0)
    g = np.where(corr_g, scan[:,:,1], 0)
    b = np.where(corr_b, scan[:,:,2], 0)


    combined = np.dstack((r,g,b))

    return combined


def main():
    #scan_dirs = get_scan_paths("double-stereo-scan-v1")
    pbr_dir = os.path.join("test-sets", "test-color-pbr")
    spec_dir = os.path.join("test-sets", "test-color-specular")
    blur_dir = os.path.join("test-sets", "test-color-blurry")
    scan_dirs = get_scan_paths(blur_dir)

    for scan_dir in scan_dirs:
        print(scan_dir)
        corr_path = os.path.join(scan_dir, "corr.png")
        proj_path = os.path.join(scan_dir, "proj.png")
        img_l_path = os.path.join(scan_dir, "img_l.png")

        corr = cv2.cvtColor(cv2.imread(corr_path), cv2.COLOR_BGR2RGB)
        proj = cv2.cvtColor(cv2.imread(proj_path), cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(cv2.imread(img_l_path), cv2.COLOR_BGR2RGB)

        combined = check_rgb_overlap_corr(img_l, corr)
        combined_proj = check_rgb_overlap_corr(proj, corr)

        cv2.imwrite(os.path.join(scan_dir, "filt_rgb.png"),cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scan_dir, "filt_proj_rgb.png"),cv2.cvtColor(combined_proj, cv2.COLOR_RGB2BGR))

        



if __name__ == '__main__':
    main()
