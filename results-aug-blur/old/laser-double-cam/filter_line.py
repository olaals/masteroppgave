import cv2
import numpy as np
import matplotlib.pyplot as plt
from oa_filter import *



def test_right_line_filter():

    averaged_nonzero = cv2.imread("i4_averaged_nonzero.png", 0)
    gt = cv2.imread("i7_gt_filt.png", 0)
    #img = cv2.resize(averaged_nonzero, (200, 200))
    cv2.imshow("", averaged_nonzero)
    cv2.waitKey(0)
    right_pixels = row_wise_max_index_mask(averaged_nonzero)
    print("max right pixels")
    print(np.max(right_pixels))
    #plt.imshow(right_pixels*255)
    #plt.show()
    avg_line_width = get_average_line_width(averaged_nonzero)
    mask2 = shift_add_horizontal(right_pixels, avg_line_width)
    print("max mask 2")
    print(np.max(mask2))
    #plt.imshow(mask2*255)
    #plt.show()

    print("avg line width")
    print(avg_line_width)

    final_filtered = np.zeros_like(avg_line_width)
    final_filtered = np.where(mask2, averaged_nonzero, 0)
    gt = 3*gt
    gt = np.clip(gt, 0, 255)
    cv2.imshow("", final_filtered)
    cv2.waitKey(0)
    cv2.imshow("", gt)
    cv2.waitKey(0)

    cv2.imwrite("i8-filtered.png", final_filtered)


    stacked = np.dstack(((final_filtered>10)*255, (gt>10)*255, np.zeros_like(final_filtered))).astype(np.uint8)
    cv2.imwrite("i9-stacked-filtered-gt.png", stacked)

    cv2.imshow("", stacked)
    cv2.waitKey(0)





    






if __name__ == '__main__':
    test_right_line_filter()

