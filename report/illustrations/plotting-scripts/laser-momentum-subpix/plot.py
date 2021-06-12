import cv2
import matplotlib.pyplot as plt
from oa_filter import *
from oa_ls import *

def main():
    img = cv2.imread("full-laser.jpg")
    cv2.imshow("",img)
    cv2.waitKey(0)
    height,width, ch = img.shape
    print(height, width)
    img = img[990:1010, 485+2:505+2]
    img = filter_value(img, 15)
    img = (img*1.5).astype(np.uint8)
    img_w = cv2.resize(img, (512, 512), 0, 0, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite("zoomed-laser.png", img_w)
    cv2.imshow("",img)
    cv2.waitKey(0)
    imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height,width = imgg.shape





    row_index = int(height//2-1)
    row = imgg[row_index, :]
    subpix = secdeg_momentum_subpix(imgg)-0.5
    x_ic = subpix[row_index]
    print(subpix)
    xs = list(range(len(row)))
    print("x_ic", x_ic)
    print(xs)
    plt.grid()
    #row[row<17] = 0
    plt.scatter(xs,row, c='red', label="Pixel values")
    plt.axvline(x_ic, label="Weighted centre of mass")
    plt.legend()
    plt.xlabel("Image x index")
    plt.ylabel("Pixel value")
    plt.show()
    

    



if __name__ == '__main__':
    main()
