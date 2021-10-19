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



def main():
    scan_dirs = get_scan_paths("test-sets/test-color-blurry")
    for scan_dir in scan_dirs:
        print(scan_dir)
        gt_scan_path = os.path.join(scan_dir, "gt_scan.png")
        gt_scan = cv2.imread(gt_scan_path, 0)
        gt_mask_path = os.path.join(scan_dir, "gt_proj_mask.png")
        gt_mask = cv2.imread(gt_mask_path, 0)

        new_mask = gt_scan>20
        new_mask = (np.bitwise_and(new_mask, gt_mask>0)*255).astype(np.uint8)

        cv2.imwrite(os.path.join(scan_dir, "gt_proj_mask.png"), new_mask)



        



if __name__ == '__main__':
    main()
