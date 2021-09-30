import cv2
import matplotlib.pyplot as plt
import numpy as np


l_img = cv2.imread("img_l.png", 0)
proj = cv2.imread("proj.png", 0)


stacked = np.dstack((l_img, proj, np.zeros_like(l_img))).astype(np.uint16)
stacked = stacked*5
stacked = np.clip(stacked, 0, 255)
stacked = stacked.astype(np.uint8)


cv2.imwrite("combined.png", cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))

