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
    kitti_parser.add_argument("--seq",type=int,default=4,choices=[i for i in range(11)])
    kitti_parser.add_argument("--index_i",type=int,default=0)
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--skip_pts",type=int,default=1)
    io_parser.add_argument("--gt_TCL_file",type=str,default="../KITTI-{:02d}/calib_res/gt_calib_{:02d}.txt")
    io_parser.add_argument("--pred_TCL_file",type=str,default="../KITTI-{:02d}/calib_res/iba_global_pl_{:02d}.txt")
    io_parser.add_argument("--dpi",type=int,default=100)
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--view",type=str2bool,default=False)
    arg_parser.add_argument("--save_path",type=str,default="../fig/04/proj_04_combine.png")
    args = parser.parse_args()
    args.seq_id = "%02d"%args.seq
    if args.skip_pts <= 0:
        args.skip_pts = 1
    args.gt_TCL_file = args.gt_TCL_file.format(args.seq, args.seq)
    args.pred_TCL_file = args.pred_TCL_file.format(args.seq, args.seq)
    return args

if __name__ == "__main__":
    args = options()
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    # extran = calibStruct.T_cam0_velo
    extran_raw = np.loadtxt(args.pred_TCL_file)
    extran = np.eye(4)
    extran[:3,:] = extran_raw[:12].reshape(3,4)
    gt_extran_raw = np.loadtxt(args.gt_TCL_file)
    gt_extran = np.eye(4)
    gt_extran[:3,:] = gt_extran_raw[:12].reshape(3,4)
    src_pcd_arr = dataStruct.get_velo(args.index_i)[::args.skip_pts,:3]  # [N, 3]
    src_img = np.array(dataStruct.get_cam0(args.index_i))  # [H, W, 3]
    img_shape = src_img.shape[:2]
    intran = calibStruct.K_cam0
    plt.figure(figsize=(12,7.6),dpi=args.dpi,tight_layout=True)
    plt.subplot(2,1,1)
    proj_src, src_rev_idx = npproj(src_pcd_arr, extran, intran, img_shape)
    range_src = src_pcd_arr[:,-1][src_rev_idx]
    plt.imshow(src_img,cmap="Greys_r")
    plt.scatter(proj_src[:,0],proj_src[:,1],marker='.',s=2.25,c=range_src,cmap='rainbow_r',alpha=0.8)
    plt.axis([0,img_shape[1],img_shape[0],0])
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplot(2,1,2)
    proj_src, src_rev_idx = npproj(src_pcd_arr, gt_extran, intran, img_shape)
    range_src = src_pcd_arr[:,-1][src_rev_idx]
    plt.imshow(src_img,cmap="Greys_r")
    plt.scatter(proj_src[:,0],proj_src[:,1],marker='.',s=2.25,c=range_src,cmap='rainbow_r',alpha=0.8)
    plt.axis([0,img_shape[1],img_shape[0],0])
    plt.axis('off')
    plt.tight_layout(pad=0)
    if args.view:
        plt.show()
    plt.savefig(args.save_path)
    

    