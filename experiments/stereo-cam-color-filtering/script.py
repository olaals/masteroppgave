from oa_blender import *
from oa_luxcore import *
import numpy as np
from oa_pointcloud_utils import *
from oa_luxcore_materials import *
from oa_bl_dataset_utils import *
import math
import matplotlib.pyplot as plt
from oa_filter import *
from oa_dev import *
import cv2

delete_all()
luxcore_setup(5)

set_hdri_luxcore("machine_shop_02_1k.hdr", gain=0.0, z_rot=2.32)
corner = import_stl("mesh646.stl")
assign_alu_low_matte(corner, 0.001, 0.001, 0.02)


resolutions=[(1080,1080), (1080, 1080), (1080, 1080)]
lumens = 5000
intra_axial_dists = [0.2,0.12]
ls = LuxcoreStereoLaserScanner("ls", location=(0.4,-0.4,0.5), resolutions=resolutions, intra_axial_dists=intra_axial_dists, z_dist=0.2, z_angle=-0.2, lumens=lumens)
ls.look_at((0,0,0))

laser_colors = [(255,0,0), (0,255,0), (0,0,255)]
ls.laser.set_laser_image_periodical(laser_colors, 20)

gt = ls.get_ground_truth_scan()

H = ls.get_planar_homography()
print(H)

r_img = ls.camera_right.get_image(halt_time=5, exposure=-8)
l_img = ls.camera_left.get_image(halt_time=5, exposure=-8)


projected_img = cv2.warpPerspective(r_img, H, ls.camera_left.resolution)

fig,ax = plt.subplots(1,4)
ax[0].imshow(l_img)
ax[1].imshow(r_img)
ax[2].imshow(projected_img)
ax[3].imshow(gt)
plt.show()
