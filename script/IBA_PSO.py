import pykitti
import numpy as np
import argparse
import os
from cv_tools import *
from scipy.spatial import cKDTree
from joblib import delayed, Parallel
from GPSOCO import PSOCO
import json
os.chdir(os.path.abspath(os.path.dirname(__file__)))

class BALoss:
    def __init__(self, intran:np.ndarray, img_shape:tuple, pointcloud_list:list, src_keypoint_list:list, tgt_keypoint_list:list, relpose_list:list,
                 offset=-1, max_pixel_dist=1.5):
        """Construct BALoss

        Args:
            intran (np.ndarray): 3x3
            img_shape (tuple): HxW
            pointcloud_list (list): list of np.ndarray Nx3
            src_keypoint_list (list): list of np.ndarray Nx2
            tgt_keypoint_list (list): list of np.ndarray Nx2
            relpose_list (list): list of np.ndarray 4x4
            offset (int, optional): error offset. Defaults to -1.
            max_pixel_dist (float, optional): maximum pixel distance between projected points and keypoints. Default to 1.5
        """
        self.intran = intran
        self.img_shape = img_shape
        self.pointcloud_list = pointcloud_list
        self.src_keypoint_list = src_keypoint_list
        self.tgt_keypoint_list = tgt_keypoint_list
        self.relpose_list = relpose_list
        self.offset = offset
        self.max_pixel_dist = max_pixel_dist
    
    def compute_error(self, sim3_log:np.ndarray):
        extran = toMat(sim3_log[:3], sim3_log[3:6])
        scale = sim3_log[-1]
        error = 0
        for pointcloud, src_keypts, tgt_keypts, relpose in zip(self.pointcloud_list, self.src_keypoint_list, self.tgt_keypoint_list, self.relpose_list):
            src_pcd_camcoord = nptran(pointcloud, extran)
            proj_src_pcd, src_rev = npproj(src_pcd_camcoord, np.eye(4), intran, self.img_shape)
            src_2d_kdtree = cKDTree(proj_src_pcd, leafsize=30)
            src_dist, src_pcd_query = src_2d_kdtree.query(src_keypts, 1, eps=1e-4, p=1)
            src_dist_rev = src_dist < self.max_pixel_dist
            src_pcd_query = src_pcd_query[src_dist_rev]
            src_matched_pts = src_keypts[src_dist_rev]
            tgt_matched_pts = tgt_keypts[src_dist_rev]
            sim3_relpose = np.copy(relpose)
            sim3_relpose[:3,3] *= scale
            tgt_pcd_camcoord = nptran(src_pcd_camcoord[src_rev][src_pcd_query], sim3_relpose)
            proj_tgt_pcd, tgt_rev = npproj(tgt_pcd_camcoord, np.eye(4), intran, self.img_shape)
            src_matched_pts = src_matched_pts[tgt_rev]
            tgt_matched_pts = tgt_matched_pts[tgt_rev]
            error += np.mean(np.sqrt(np.sum((tgt_matched_pts - proj_tgt_pcd)**2,axis=1)))
        return error/len(self.relpose_list)
    
    def vis_error(self, sim3_log:np.ndarray, index:int):
        extran = toMat(sim3_log[:3], sim3_log[3:6])
        scale = sim3_log[-1]
        pointcloud = self.pointcloud_list[index]
        src_keypts = self.src_keypoint_list[index]
        tgt_keypts = self.tgt_keypoint_list[index]
        relpose = self.relpose_list[index]
        src_pcd_camcoord = nptran(pointcloud, extran)
        proj_src_pcd, src_rev = npproj(src_pcd_camcoord, np.eye(4), intran, self.img_shape)
        src_2d_kdtree = cKDTree(proj_src_pcd, leafsize=30)
        src_dist, src_pcd_query = src_2d_kdtree.query(src_keypts, 1, eps=1e-4, p=1)
        src_dist_rev = src_dist < self.max_pixel_dist
        src_pcd_query = src_pcd_query[src_dist_rev]
        src_matched_pts = src_keypts[src_dist_rev]
        tgt_matched_pts = tgt_keypts[src_dist_rev]
        sim3_relpose = np.copy(relpose)
        sim3_relpose[:,:3] *= scale
        tgt_pcd_camcoord = nptran(src_pcd_camcoord[src_rev][src_pcd_query], sim3_relpose)
        proj_tgt_pcd, tgt_rev = npproj(tgt_pcd_camcoord, np.eye(4), intran, self.img_shape)
        src_matched_pts = src_matched_pts[tgt_rev]
        tgt_matched_pts = tgt_matched_pts[tgt_rev]
        return src_matched_pts, tgt_matched_pts, proj_tgt_pcd

    def compute_error_vec(self, sim3_log_vec:np.ndarray):
        parallel_res = Parallel(n_jobs=-1)(delayed(self.compute_error)(sim3_log) for sim3_log in sim3_log_vec)
        return np.array(parallel_res)

def str2bool(s:str) -> bool:
    if s.isdigit():
        if float(s) > 0:
            return True
        else:
            return False
    if s.lower() == "false":
        return False
    else:
        return True

def options():
    parser = argparse.ArgumentParser()
    kitti_parser = parser.add_argument_group()
    kitti_parser.add_argument("--base_dir",type=str,default="/data/DATA/data_odometry/dataset/")
    kitti_parser.add_argument("--seq",type=int,default=0,choices=[i for i in range(11)])
    kitti_parser.add_argument("--img_shape", type=int, nargs=2, default=[376, 1241])
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--keypoint_dir",type=str,default="../debug/data/")
    io_parser.add_argument("--init_sim3",type=str,default="../KITTI-00/calib_res/he_calib_00.txt")
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--offset",type=float,default=-1)
    arg_parser.add_argument("--max_pixel_dist",type=float,default=3)
    arg_parser.add_argument("--range_x",type=float,nargs=7, default=[0.1,0.1,0.1,0.4,0.4,0.4,1.0])
    arg_parser.add_argument("--particle_size",type=int,default=4000)
    arg_parser.add_argument("--max_iter",type=int,default=100)
    arg_parser.add_argument("--vratio",type=float,default=0.1)
    arg_parser.add_argument("--view",type=str2bool,default=True)
    args = parser.parse_args()
    args.seq_id = "%02d"%args.seq
    return args


if __name__ == "__main__":
    np.random.seed(0)  # color consistency
    args = options()
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    extran = calibStruct.T_cam0_velo  # [4,4]
    intran = calibStruct.K_cam0   # [3, 3]
    init_sim3_data = np.loadtxt(args.init_sim3)
    init_se3_data = init_sim3_data[:12]
    init_sim3_log = np.zeros(7)
    se3_mat = np.eye(4)
    se3_mat[:3,:] = init_se3_data.reshape(3,4)
    scale = init_sim3_data[-1]
    rotvec, tsl = toVec(se3_mat)
    init_sim3_log[:3] = rotvec
    init_sim3_log[3:6] = tsl 
    init_sim3_log[-1] = scale
    keypts_files = list(sorted(os.listdir(args.keypoint_dir)))
    print("Load %d Keypoint files."%len(keypts_files))
    pointcloud_list = []
    src_keypoint_list = []
    tgt_keypoint_list = []
    relpose_list = []
    for file in keypts_files:
        full_path = os.path.join(args.keypoint_dir, file)
        data = json.load(open(full_path,'r'))
        src_idx = data["src_mnFrameId"]  # file index
        src_pose = np.array(data["src_pose"])  # Tc1w
        tgt_pose = np.array(data["tgt_pose"])  # Tc2w
        relpose_list.append(tgt_pose @ inv_pose(src_pose))
        pointcloud_list.append(dataStruct.get_velo(src_idx)[:,:3])
        src_keypoint_list.append(np.array(data["src_matched_keypoint"]))
        tgt_keypoint_list.append(np.array(data["tgt_matched_keypoint"]))
    
    Xrange = np.array(args.range_x)
    Xmin = init_sim3_log - Xrange
    Xmax = init_sim3_log + Xrange
    Loss = BALoss(intran, args.img_shape, pointcloud_list, src_keypoint_list, tgt_keypoint_list, relpose_list, args.offset, args.max_pixel_dist)
    print("init_sim3:{}".format(init_sim3_log))
    print("init error:{}".format(Loss.compute_error(init_sim3_log)))
    pso = PSOCO(
        args.particle_size,
        args.max_iter,
        7,
        Loss.compute_error_vec,
        Xmin = Xmin,
        Xmax = Xmax,
        Vmin = -Xrange * args.vratio,
        Vmax = Xrange * args.vratio,
        record_T = -1,
        hybrid = False
    )
    pso.init_Population(init_sim3_log)
    sol,fval = pso.solve()
    print("Pg:{},fval:{}".format(sol, fval))
    sol_extran = toMat(sol[:3],sol[3:6])
    scale = sol[-1]
    print("solved extran:{}".format(sol_extran))
    print("solved scale:{}".format(scale))