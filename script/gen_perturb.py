import argparse
import numpy as np
from cv_tools import toVec, toVecSplit
from scipy.spatial.transform import Rotation
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_sim3",type=str,default="../KITTI-04/calib_res/gt_calib_04.txt")
    parser.add_argument("--rot_step",type=float,default=1)
    parser.add_argument("--rot_range",type=float,default=10)
    parser.add_argument("--tsl_step",type=float,default=0.01)
    parser.add_argument("--tsl_range",type=float,default=0.1)
    parser.add_argument("--scale_step",type=float,default=0.1)
    parser.add_argument("--scale_range",type=float,default=1)
    parser.add_argument("--save_sim3",type=str,default="../demo/sim3_test_04.txt")
    parser.add_argument("--log_sim3",type=str,default="../demo/sim3_log_04.txt")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    rot_offset = np.arange(-args.rot_range, args.rot_range+1e-8, args.rot_step)
    tsl_offset = np.arange(-args.tsl_range, args.tsl_range+1e-8, args.tsl_step)
    scale_offset = np.arange(-args.scale_range, args.scale_range+1e-8, args.scale_step)
    base_sim3_data = np.loadtxt(args.base_sim3)
    base_rigid_mat = np.eye(4)
    base_rigid_mat[:3,:] = base_sim3_data[:12].reshape(3,4)
    base_scale = base_sim3_data[-1]
    base_rot = base_rigid_mat[:3,:3]
    base_tsl = base_rigid_mat[:3,3]
    baseR = Rotation.from_matrix(base_rot)
    base_rpy = baseR.as_euler("zyx",degrees=True)
    log_f = open(args.log_sim3,'w')
    save_f = open(args.save_sim3, 'w')
    base_xvec = np.zeros(7)
    base_xvec[:3], base_xvec[3:6] = toVec(base_rigid_mat)
    base_xvec[-1] = base_scale
    # log_f.write("Raw data\n")
    # save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*base_xvec))
    # roll
    for droll in rot_offset:
        rpy = base_rpy + np.array([droll,0,0])
        xvec = np.copy(base_xvec)
        xvec[:3], xvec[3:6] = toVecSplit(Rotation.from_euler("zyx",rpy,True).as_matrix(), base_tsl)
        log_f.write("Roll offset: %0.6f\n"%droll)
        save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*xvec))
    # pitch
    for dptich in rot_offset:
        rpy = base_rpy + np.array([0,dptich,0])
        xvec = np.copy(base_xvec)
        xvec[:3], xvec[3:6] = toVecSplit(Rotation.from_euler("zyx",rpy,True).as_matrix(), base_tsl)
        log_f.write("Pitch offset: %0.6f\n"%dptich)
        save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*xvec))
    # yaw
    for dyaw in rot_offset:
        rpy = base_rpy + np.array([0,0,dyaw])
        xvec = np.copy(base_xvec)
        xvec[:3], xvec[3:6] = toVecSplit(Rotation.from_euler("zyx",rpy,True).as_matrix(), base_tsl)
        log_f.write("Yaw offset: %0.6f\n"%dyaw)
        save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*xvec))
    # x
    for dx in tsl_offset:
        tsl = base_tsl + np.array([dx,0,0])
        xvec = np.copy(base_xvec)
        xvec[:3], xvec[3:6] = toVecSplit(base_rot, tsl)
        log_f.write("X offset: %0.6f\n"%dx)
        save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*xvec))
    # y
    for dy in tsl_offset:
        tsl = base_tsl + np.array([0,dy,0])
        xvec = np.copy(base_xvec)
        xvec[:3], xvec[3:6] = toVecSplit(base_rot, tsl)
        log_f.write("Y offset: %0.6f\n"%dy)
        save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*xvec))
    # z
    for dz in tsl_offset:
        tsl = base_tsl + np.array([0,0,dz])
        xvec = np.copy(base_xvec)
        xvec[:3], xvec[3:6] = toVecSplit(base_rot, tsl)
        log_f.write("Z offset: %0.6f\n"%dz)
        save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*xvec))
    # scale
    for i,ds in enumerate(scale_offset):
        xvec = np.copy(base_xvec)
        xvec[-1] += ds
        if i != scale_offset.size -1:
            log_f.write("scale offset: %0.6f\n"%ds)
            save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(*xvec))  
        else:
            log_f.write("scale offset: %0.6f"%ds)
            save_f.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}".format(*xvec))  
    log_f.close()
    save_f.close()
    
    