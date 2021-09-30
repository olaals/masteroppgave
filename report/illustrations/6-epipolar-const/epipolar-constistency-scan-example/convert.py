import cv2
import matplotlib.pyplot as plt

def split_rgb(img):
    red_img_l = img.copy()
    red_img_l[:,:,1] = 0
    red_img_l[:,:,2] = 0
    green_img_l = img.copy()
    green_img_l[:,:,0] = 0
    green_img_l[:,:,2] = 0
    blue_img_l = img.copy()
    blue_img_l[:,:,0] = 0
    blue_img_l[:,:,1] = 0
    return red_img_l, green_img_l, blue_img_l


img = cv2.cvtColor(cv2.imread("filt_rgb.png"), cv2.COLOR_BGR2RGB)
corr = cv2.cvtColor(cv2.imread("corr.png"), cv2.COLOR_BGR2RGB)

red_img_l, green_img_l, blue_img_l = split_rgb(img)
red_corr, green_corr, blue_corr = split_rgb(corr)

red_img_l_gray = np.max(red_img_l, axis=2)
green_img_l_gray = np.max(red_img_l, axis=2)
blue_img_l_gray = np.max(red_img_l, axis=2)


fig,ax = plt.subplots(1,3)
ax[0].imshow(red_img_l)
ax[1].imshow(green_img_l)
ax[2].imshow(blue_img_l)
plt.show()

fig,ax = plt.subplots(1,3)
ax[0].imshow(red_corr)
ax[1].imshow(green_corr)
ax[2].imshow(blue_corr)
plt.show()

cv2.imwrite("red_img_l.png", cv2.cvtColor(red_img_l, cv2.COLOR_RGB2BGR))
cv2.imwrite("green_img_l.png", cv2.cvtColor(green_img_l, cv2.COLOR_RGB2BGR))
cv2.imwrite("blue_img_l.png", cv2.cvtColor(blue_img_l, cv2.COLOR_RGB2BGR))

cv2.imwrite("red_corr.png", cv2.cvtColor(red_corr, cv2.COLOR_RGB2BGR))
cv2.imwrite("green_corr.png", cv2.cvtColor(green_corr, cv2.COLOR_RGB2BGR))
cv2.imwrite("blue_corr.png", cv2.cvtColor(blue_corr, cv2.COLOR_RGB2BGR))

