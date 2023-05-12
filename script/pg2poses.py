import numpy as np
import open3d as o3d

pg_path = "/home/bit/CODE/Research/AutoCalib/ST_Calib/floam_optpg_00.json"
pose_path = "/home/bit/CODE/Research/AutoCalib/ST_Calib/floam_opt_00.txt"
pg = o3d.io.read_pose_graph(pg_path)
pg_arr = np.zeros([len(pg.nodes),12])
for i in range(len(pg.nodes)):
    pg_arr[i] = pg.nodes[i].pose[:3,:].flatten()
np.savetxt(pose_path, pg_arr)