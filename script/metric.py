import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import os

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",type=str,default="../KITTI-02/calib_res/gt_calib_02.txt")
    parser.add_argument("--pred",type=str,default="../KITTI-02/calib_res/iba_global_bias_02.txt")
    return parser.parse_args()

def inv_pose(pose:np.ndarray):
    ivpose = np.eye(4)
    ivpose[:3,:3] = pose[:3,:3].T
    ivpose[:3,3] = -ivpose[:3,:3] @ pose[:3, 3]
    return ivpose

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    args = options()
    gt_data = np.loadtxt(args.gt)[:12]
    pred_data = np.loadtxt(args.pred)[:12]
    gt_ex = np.eye(4)
    gt_ex[:3,:] = gt_data.reshape(3,4)
    pred_ex = np.eye(4)
    pred_ex[:3,:] = pred_data.reshape(3,4)
    err_ex = inv_pose(gt_ex) @ pred_ex
    tsl_err = err_ex[:3,3]
    R = Rotation.from_matrix(err_ex[:3,:3])
    rot_err = R.as_euler("zyx",True)
    print("x:{},y:{},z:{}".format(*tsl_err))
    print("roll:{},pitch:{},yaw:{}".format(*rot_err))