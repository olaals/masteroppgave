from oa_blender import *
from oa_luxcore import *
import numpy as np
from oa_pointcloud_utils import *
from oa_bl_dataset_utils import *
import math

delete_all()
luxcore_setup(5)

 
import_stl("mesh545.stl")

ls = LuxcoreLaserScanner("ls", location=(0.8,-0.3,0.8), laser_resolution=(501, 4000), laser_sensor_width=4)
ls.look_at((0,0,0))


