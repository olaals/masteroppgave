import cv2
import matplotlib.pyplot as plt
import numpy as np
from oa_dev import *



gt = cv2.imread("gt.png", 0)
pred = cv2.imread("pred.png", 0)


stacked = np.dstack((pred, gt, np.zeros_like(gt)))

cv2_imwrite("stacked.png", stacked)



fig,ax = plt.subplots(1,3)
ax[0].imshow(gt)
ax[1].imshow(pred)
ax[2].imshow(stacked)
plt.show()
