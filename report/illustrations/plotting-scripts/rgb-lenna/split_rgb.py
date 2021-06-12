import cv2
import matplotlib.pyplot as plt



lenna = cv2.imread("lenna.png")
lenna = cv2.cvtColor(lenna, cv2.COLOR_BGR2RGB)


r = lenna.copy()
r[:,:,1] = 0
r[:,:,2] = 0

g = lenna.copy()
g[:,:,0] = 0
g[:,:,2] = 0

b = lenna.copy()
b[:,:,0] = 0
b[:,:,1] = 1


fig,ax = plt.subplots(1,3)
ax[0].imshow(r)
ax[1].imshow(g)
ax[2].imshow(b)
plt.show()




