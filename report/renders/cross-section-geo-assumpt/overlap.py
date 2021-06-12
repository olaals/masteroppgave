import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


scandirs = ["scan0078", "scan0081", "scan0084", "scan0088"]


i = 0
for scandir in scandirs:

    img_l_path = os.path.join(scandir, "img_l.png")
    proj_path = os.path.join(scandir, "proj.png")
    gt_path = os.path.join(scandir, "gt_scan.png")
    non_avg_path = os.path.join(scandir, "nonzero_avg.png")

    gt = cv2.imread(gt_path,0)
    nonavg = cv2.imread(non_avg_path, 0)
    img_l = cv2.imread(img_l_path, 0)
    proj = cv2.imread(proj_path, 0)
    
    if i == 1:
        stacked = np.dstack((gt, nonavg, np.zeros_like(gt)))[200:950,350:750,:]
        stacked_refl = np.dstack((img_l, proj, np.zeros_like(gt)))[200:950,350:750,:]
    else:
        stacked = np.dstack((gt, nonavg, np.zeros_like(gt)))[200:950,250:650,:]
        stacked_refl = np.dstack((img_l, proj, np.zeros_like(gt)))[200:950,250:650,:] 

    cv2.imwrite("gt_stack"+str(i)+".png", cv2.cvtColor(stacked*4, cv2.COLOR_RGB2BGR))
    cv2.imwrite("pred_stack"+str(i)+".png", cv2.cvtColor(stacked_refl*4, cv2.COLOR_RGB2BGR))
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(stacked_refl*4)
    ax[1].imshow(stacked*4)
    plt.show()
    i+=1


