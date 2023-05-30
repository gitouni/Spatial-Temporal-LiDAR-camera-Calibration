import pykitti
import numpy as np
import argparse
import os
import cv2
# import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from cv_tools import *
from scipy.spatial.ckdtree import cKDTree
from fps_v1 import FPS


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
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--KeyFrameDir",type=str,default="../KITTI-00/KeyFrames")
    io_parser.add_argument("--MapFile",type=str,default="../KITTI-00/Map.xml")
    io_parser.add_argument("--KeyPointsKey",type=str,default="mvKeysUn")
    io_parser.add_argument("--MapPointKey",type=str,default="mvpMapPointsId")
    io_parser.add_argument("--index_i",type=int,default=0)
    io_parser.add_argument("--index_j",type=int,default=3)
    io_parser.add_argument("--debug_log",type=str,default="")
    io_parser.add_argument("--Twc_file",type=str,default="../Twc.txt")
    io_parser.add_argument("--Twl_file",type=str,default="../Twl.txt")
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--tsl_perturb",type=float,nargs=3,default=[0.1,-0.15,0.1])
    arg_parser.add_argument("--rot_perturb",type=float,nargs=3,default=[0,0,0])
    arg_parser.add_argument("--fps_sample",type=int,default=-1)
    arg_parser.add_argument("--pixel_corr_dist",type=float,default=3)
    arg_parser.add_argument("--view",type=str2bool,default=True)
    args = parser.parse_args()
    args.seq_id = "%02d"%args.seq
    return args


def drawcorrpoints(img1,img2,pts1,pts2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def draw3corrpoints(img1,img2,pts11,pts12,pts2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for pt11,pt12,pt2 in zip(pts11,pts12,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv2.circle(img1,tuple(pt11),5,color,1)
        img1 = cv2.circle(img1,tuple(pt11),3,[255,255,255],1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,1)
    return img1,img2

def getMatchedId(mapnode, src_map_ptid:np.ndarray, tgt_map_ptid:np.ndarray, src_index:int, tgt_index:int):
    src_in_tgt = np.isin(src_map_ptid, tgt_map_ptid)
    src_inlier_idx = src_map_ptid[src_in_tgt]  # Common MapPoint
    src_kpt_idx = []
    tgt_kpt_idx = []
    for idx in src_inlier_idx:
        mappt_node_name = "MapPoint_{:d}".format(idx)
        mappt_node = mapnode.getNode(mappt_node_name)
        mappt_kfid_node = mappt_node.getNode("mobsMapKFId")
        mappt_kpid_node = mappt_node.getNode("mMapKFInId")
        mappt_kfid = [int(mappt_kfid_node.at(i).real()) for i in range(mappt_kfid_node.size())]
        mappt_kpid = [int(mappt_kpid_node.at(i).real()) for i in range(mappt_kpid_node.size())]
        if src_index in mappt_kfid and tgt_index in mappt_kfid:
            src_kfid_in_mappt = mappt_kfid.index(src_index)
            tgt_kfid_in_mappt = mappt_kfid.index(tgt_index)
            src_kpt_idx.append(mappt_kpid[src_kfid_in_mappt])
            tgt_kpt_idx.append(mappt_kpid[tgt_kfid_in_mappt])
    return src_kpt_idx, tgt_kpt_idx
        

def compute_orb_desc(img:np.ndarray):
    """compute orb descriptors

    Args:
        img (np.ndarray): HxW

    Returns:
        _type_: keypoints, descriptors
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp, desc = orb.detectAndCompute(img, None)
    return kp, desc

if __name__ == "__main__":
    np.random.seed(0)  # color consistency
    args = options()
    augT = toMat(args.rot_perturb, args.tsl_perturb)
    print("augT:\n{}".format(augT))
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    extran = calibStruct.T_cam0_velo  # [4,4]
    print("GT TCL:\n{}".format(extran))
    print("GT TCL Rvec:{}\ntvec:{}".format(*toVec(extran)))
    KeyFramesFiles = list(sorted(os.listdir(args.KeyFrameDir)))
    KeyFramesFiles = [file for file in KeyFramesFiles if os.path.splitext(file)[1] == '.yml']
    src_KeyFrameFile = os.path.join(args.KeyFrameDir, KeyFramesFiles[args.index_i])
    tgt_KeyFrameFile = os.path.join(args.KeyFrameDir, KeyFramesFiles[args.index_j])
    src_fs = cv2.FileStorage(src_KeyFrameFile, cv2.FILE_STORAGE_READ)
    tgt_fs = cv2.FileStorage(tgt_KeyFrameFile, cv2.FILE_STORAGE_READ)
    map_fs = cv2.FileStorage(args.MapFile, cv2.FILE_STORAGE_READ)
    src_kptnode = src_fs.getNode(args.KeyPointsKey)
    tgt_kptnode = tgt_fs.getNode(args.KeyPointsKey)
    src_keypts = []
    tgt_keypts = []
    for i in range(src_kptnode.size()):
        src_keypts.append([int(src_kptnode.at(i).at(0).real()), int(src_kptnode.at(i).at(1).real())])
    for i in range(tgt_kptnode.size()):
        tgt_keypts.append([int(tgt_kptnode.at(i).at(0).real()), int(tgt_kptnode.at(i).at(1).real())])
    src_mapptnode = src_fs.getNode(args.MapPointKey)
    tgt_mapptnode = tgt_fs.getNode(args.MapPointKey)
    src_map_ptid = []
    tgt_map_ptid = []
    for i in range(src_mapptnode.size()):
        src_map_ptid.append(int(src_mapptnode.at(i).real()))
    for i in range(tgt_mapptnode.size()):
        tgt_map_ptid.append(int(tgt_mapptnode.at(i).real()))
    mapnode = map_fs.getNode("mspMapPoints")
      
    print("src KeyPoints:{} | src KeyPoints: {}".format(len(src_keypts), len(src_map_ptid)))
    print("tgt MapPoints:{} | tgt MapPoints: {}".format(len(tgt_keypts), len(tgt_map_ptid)))
    src_keypts, tgt_keypts, src_map_ptid, tgt_map_ptid = list(map(lambda arr:np.array(arr),[src_keypts, tgt_keypts, src_map_ptid, tgt_map_ptid]))
    aug_extran = augT @ extran
    intran = calibStruct.K_cam0
    src_pcd_arr = dataStruct.get_velo(args.index_i)[:,:3]  # [N, 3]
    tgt_pcd_arr = dataStruct.get_velo(args.index_j)[:,:3]  # [N, 3]
    src_img = np.array(dataStruct.get_cam0(args.index_i))  # [H, W, 3]
    tgt_img = np.array(dataStruct.get_cam0(args.index_j))  # [H, W, 3]
    src_pose = dataStruct.poses[args.index_i]
    tgt_pose = dataStruct.poses[args.index_j]
    camera_motion:np.ndarray = inv_pose(tgt_pose) @ src_pose  # Tc2w * Twc1
    if(args.debug_log == ""):
        print("Compute Keypoint Matching from Map")
        src_matched_pts_idx, tgt_matched_pts_idx = getMatchedId(mapnode, src_map_ptid, tgt_map_ptid, args.index_i, args.index_j)
        src_matched_pts_idx, tgt_matched_pts_idx = np.array(src_matched_pts_idx), np.array(tgt_matched_pts_idx)
    else:
        print("Load Keypoint Matching from {}".format(args.debug_log))
        matched_pts_idx = np.loadtxt(args.debug_log,dtype=np.int32)
        src_matched_pts_idx, tgt_matched_pts_idx = matched_pts_idx[:,0], matched_pts_idx[:,1]
    src_matched_pts = src_keypts[src_matched_pts_idx]
    tgt_matched_pts = tgt_keypts[tgt_matched_pts_idx]
    print("Matched Keypoints:{}".format(len(src_matched_pts)))
    src_pcd_camcoord = nptran(src_pcd_arr, extran)
    proj_src_pcd, src_rev = npproj(src_pcd_camcoord, np.eye(4), intran, src_img.shape)
    src_2d_kdtree = cKDTree(proj_src_pcd, leafsize=30)
    src_dist, src_pcd_query = src_2d_kdtree.query(src_matched_pts, 1, eps=1e-4, p=1)
 
    src_dist_rev = src_dist < args.pixel_corr_dist
    src_matched_pts = src_matched_pts[src_dist_rev]
    tgt_matched_pts = tgt_matched_pts[src_dist_rev]
    src_pcd_query = src_pcd_query[src_dist_rev]

    src2tgt_pcd_arr = nptran(src_pcd_camcoord[src_rev][src_pcd_query], camera_motion)
    src2tgt_proj_pcd, src2tgt_proj_rev = npproj(src2tgt_pcd_arr, np.eye(4), intran, tgt_img.shape)
    src_matched_pts = src_matched_pts[src2tgt_proj_rev]
    tgt_matched_pts = tgt_matched_pts[src2tgt_proj_rev]
    tgt_matched_pts, src2tgt_proj_pcd, src_matched_pts = list(map(lambda arr: arr.astype(np.int32),[tgt_matched_pts, src2tgt_proj_pcd, src_matched_pts]))
    if(args.fps_sample > 0):
        fps = FPS(tgt_matched_pts, n_samples=args.fps_sample)
        _, fps_idx = fps.fit(True)
    else:
        fps_idx = np.arange(tgt_matched_pts.shape[0])
    # draw_src_img, draw_tgt_img = drawcorrpoints(src_img, tgt_img, src_matched_pts[fps_idx], tgt_matched_pts[fps_idx])
    draw_tgt_img, draw_src_img = draw3corrpoints(tgt_img, src_img, tgt_matched_pts[fps_idx], src2tgt_proj_pcd[fps_idx], src_matched_pts[fps_idx])
    plt.figure(dpi=200)
    plt.subplot(2,1,1)
    plt.title("Matched points:{}".format(tgt_matched_pts.shape[0]))
    plt.imshow(draw_src_img)
    plt.subplot(2,1,2)
    plt.imshow(draw_tgt_img)
    plt.show()
    # plt.savefig("../demo_IBA_FS.pdf")
    