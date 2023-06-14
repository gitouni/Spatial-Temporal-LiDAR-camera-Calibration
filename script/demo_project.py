import pykitti
import numpy as np
import argparse
import os
# import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from cv_tools import *


os.chdir(os.path.dirname(__file__))

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
    kitti_parser.add_argument("--index_i",type=int,default=0)
    kitti_parser.add_argument("--index_j",type=int,default=3)
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--skip_pts",type=int,default=1)
    io_parser.add_argument("--index_file",type=str,default="../KITTI-00/FrameId.yml")
    io_parser.add_argument("--Twc_file",type=str,default="../KITTI-00/slam_res/Twc.txt")
    io_parser.add_argument("--Twl_file",type=str,default="../KITTI-00/slam_res/floam_isam_00.txt")
    io_parser.add_argument("--TCL_file",type=str,default="../KITTI-00/calib_res/iba_calib_00.txt")
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--view",type=str2bool,default=True)
    args = parser.parse_args()
    args.seq_id = "%02d"%args.seq
    if args.skip_pts <= 0:
        args.skip_pts = 1
    return args

if __name__ == "__main__":
    args = options()
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    extran_raw = np.loadtxt(args.TCL_file)
    extran = np.eye(4)
    extran[:3,:] = extran_raw[:12].reshape(3,4)
    scale = extran_raw[-1]
    src_pcd_arr = dataStruct.get_velo(args.index_i)[::args.skip_pts,:3]  # [N, 3]
    tgt_pcd_arr = dataStruct.get_velo(args.index_j)[::args.skip_pts,:3]  # [N, 3]
    src_img = np.array(dataStruct.get_cam2(args.index_i))  # [H, W, 3]
    tgt_img = np.array(dataStruct.get_cam2(args.index_j))  # [H, W, 3]
    # src_img = np.flip(src_img,axis=2)  # BGR to RGB
    # tgt_img = np.flip(tgt_img,axis=2)
    img_shape = src_img.shape[:2]
    intran = calibStruct.K_cam0
    proj_src, src_rev_idx = npproj(src_pcd_arr, extran, intran, img_shape)
    range_src = nptran(src_pcd_arr, extran)[:,-1][src_rev_idx]
    proj_tgt, tgt_rev_idx = npproj(src_pcd_arr, extran, intran, img_shape)
    range_tgt = nptran(tgt_pcd_arr, extran)[:,-1][tgt_rev_idx]
    tgt_in_src = np.isin(tgt_rev_idx,src_rev_idx)
    src_in_tgt = np.isin(src_rev_idx,tgt_rev_idx)
    proj_src = proj_src[tgt_in_src]
    range_src = range_src[tgt_in_src]
    proj_tgt = proj_tgt[src_in_tgt]
    range_tgt = range_tgt[src_in_tgt]
    plt.subplot(2,1,1)
    plt.imshow(src_img)
    plt.scatter(proj_src[:,0],proj_src[:,1],c=range_src,cmap='rainbow_r',alpha=0.4,s=1)
    plt.subplot(2,1,2)
    plt.imshow(tgt_img)
    plt.scatter(proj_tgt[:,0],proj_tgt[:,1],c=range_tgt,cmap='rainbow_r',alpha=0.4,s=1)
    plt.show()
    

    