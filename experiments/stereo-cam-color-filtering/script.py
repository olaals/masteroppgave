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

delete_all()
luxcore_setup(5)

 
corner = import_stl("mesh646.stl")
assign_alu_low_matte(corner, 0.001, 0.001, 0.02)

ls = LuxcoreLaserScanner("ls", location=(0.8,-0.3,0.8), camera_resolution=(1024,1024), laser_resolution=(1024, 1024), lumens=5000)

ls.look_at((0,0,0))
laser_colors = [(255,0,0), (0,255,0), (0,0,255)]
ls.laser.set_laser_image_periodical(laser_colors, 20)
gt = ls.get_ground_truth_scan()
gt_mask = gt>0






set_hdri_luxcore("machine_shop_02_1k.hdr", gain=0.4, z_rot=2.32)
norm_exp_img = ls.camera.get_image(halt_time=6, exposure=-1)
cv2_imwrite("images/1_init.png", norm_exp_img)
bpy.context.scene.world.luxcore.gain = 0.0



camimg = ls.camera.get_image(halt_time=6, exposure=-8)
gt_cam = np.where(d3stack(gt_mask), camimg, 0)

cv2_imwrite("images/0_gt_cam.png", gt_cam)

cv2_imwrite("images/2_low_exp.png", camimg)

filt_camimg = filter_value(camimg, 60)



corr_img = ls.get_laser_correspondance_img()
cv2_imwrite("images/3_correspondance_img.png", corr_img)



mask = filter_similar_hue_multicolor(filt_camimg, corr_img, laser_colors, 10, min_value=50, pad=1)
cv2_write_mask("images/4_filter_mask.png", mask)

filtered = np.where(d3stack(mask), camimg, 0)
cv2_imwrite("images/5_filtered.png", filtered)


fig,ax = plt.subplots(2,3)

ax[0,0].imshow(filt_camimg)
ax[0,1].imshow(corr_img)
ax[0,2].imshow(gt)
ax[1,0].imshow(mask)
ax[1,1].imshow(filtered)
plt.show()

