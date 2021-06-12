import open3d as o3d
import numpy as np



def visualize_pcs():
    pcd_l = o3d.io.read_point_cloud("pc_p_W.xyz")
    pcd_p = o3d.io.read_point_cloud("pc_l_W.xyz")
    pcd_l.paint_uniform_color([1, 0, 0])
    pcd_p.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd_l, pcd_p])

    o3d.io.write_triangle_mesh("pcd_p.ply", pcd_p)






if __name__ == '__main__':
    visualize_pcs()
