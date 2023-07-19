import argparse
import numpy as np
from cv_tools import toMat, toVec
from scipy.spatial.transform import Rotation
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_dir",type=str,default="../KITTI-00/calib_res")
    parser.add_argument("--gt_calib",type=str,default="gt_calib_00.txt")
    parser.add_argument("--pred_se3_calib",type=str,default="calibnet_05.txt")
    return parser.parse_args()


def inv_pose(pose:np.ndarray):
    ivpose = np.eye(4)
    ivpose[:3,:3] = pose[:3,:3].T
    ivpose[:3,3] = -ivpose[:3,:3] @ pose[:3, 3]
    return ivpose

if __name__ == "__main__":
    args = options()
    gt_extran = np.eye(4)
    gt_extran_data = np.loadtxt(os.path.join(args.calib_dir, args.gt_calib))[:12]
    gt_extran[:3,:] = gt_extran_data.reshape(3,4)
    pred_se3list = np.loadtxt(os.path.join(args.calib_dir, args.pred_se3_calib))
    best_err = np.inf
    inv_gt_extran = inv_pose(gt_extran)
    best_se3mat = np.eye(4)
    for se3 in pred_se3list:
        se3mat = toMat(se3[:3],se3[3:])
        errmat = inv_gt_extran @ se3mat
        tsl_err = np.linalg.norm(errmat[:3,3])
        R = Rotation.from_matrix(errmat[:3,:3])
        rot_err = np.linalg.norm(R.as_euler("zyx",False))
        if tsl_err + rot_err < best_err:
            best_se3mat = se3mat
            best_err = tsl_err + rot_err
    errmat = inv_gt_extran @ best_se3mat
    R = Rotation.from_matrix(errmat[:3,:3])
    rotdeg = R.as_euler("zyx",True)
    tslerr = errmat[:3,3]
    print("Roll:{} Pitch:{} Yaw:{}".format(*rotdeg))
    print("X:{} Y:{} Z:{}".format(*tslerr))
    print("Rotation RMSE:{} degree".format(np.linalg.norm(rotdeg)))
    print("Translation RMSE:{} m".format(np.linalg.norm(tslerr)))
