import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation

os.chdir(os.path.dirname(__file__))

def read_pose_file(filename:str) -> list:
    raw_data = np.loadtxt(filename)
    pose_list = []
    for line in raw_data:
        pose = np.eye(4)
        pose[:3, :] = line.reshape(3,4)
        pose_list.append(pose)
    return pose_list

def pose2motion(pose_list:list) -> list:
    motion_list = []
    for i in range(len(pose_list)-1):
        motion = pose_list[i+1] @ np.linalg.inv(pose_list[i])
        motion_list.append(motion)
    return motion_list

def toVec(Rmat:np.ndarray):
    R = Rotation.from_matrix(Rmat)
    vecR = R.as_rotvec()
    return vecR

def hand_eye_calib(camera_edge:list, pcd_edge:list):
    assert len(camera_edge) == len(pcd_edge)
    N = len(camera_edge)
    alpha = np.zeros([N,3])
    beta = alpha.copy()
    for i,(cedge,pedge) in enumerate(zip(camera_edge,pcd_edge)):
        cvec = toVec(cedge[:3,:3])
        pvec = toVec(pedge[:3,:3])
        alpha[i,:] = cvec
        beta[i,:] = pvec
    alpha -= alpha.mean(axis=0,keepdims=True)
    beta -= beta.mean(axis=0,keepdims=True)
    H = np.dot(beta.T,alpha)  # (3,3)
    U, S, Vt = np.linalg.svd(H)  
    R = np.dot(Vt.T, U.T)     
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    AA = []
    BB = []
    for cedge,pedge in zip(camera_edge,pcd_edge):
        AA.append(np.hstack((cedge[:3,:3]-np.eye(3),cedge[:3,[3]])))  # (3,4)
        BB.append(R @ pedge[:3,[3]])  # (3,1)
    AA, BB = np.vstack(AA), np.vstack(BB)  # (3N, 4), (3N, 1)
    sol = np.linalg.solve(AA.T @ AA, AA.T @ BB)
    return R, sol[:3].reshape(-1), sol[-1]  # t, s

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Twc_file",type=str,default="../Twc.txt")
    parser.add_argument("--Twl_file",type=str,default="../Twl.txt")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    Tcw_list = pose2motion(read_pose_file(args.Twc_file))
    Tlw_list = pose2motion(read_pose_file(args.Twl_file))
    R, t, s = hand_eye_calib(Tcw_list, Tlw_list)
    print("R:\n{}\nt:\n:{}\ns:\n{}".format(R,t,s))
    