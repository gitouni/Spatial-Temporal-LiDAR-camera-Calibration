import pykitti
import numpy as np
import argparse
import os
import cv2
# import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from cv_tools import *
from scipy.spatial import cKDTree
from GP_reg import GPR
from joblib import delayed, Parallel
import yaml

os.chdir(os.path.dirname(__file__))

def print_params(title:str="", params:dict={}):
    print(title,end="")
    for key, value in params.items():
        print("{}: {}".format(key, value),end=" ")
    print()

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
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--KeyFrameDir",type=str,default="../KITTI-04/KeyFrames")
    io_parser.add_argument("--FrameIdFile",type=str,default="../KITTI-04/FrameId.yml")
    io_parser.add_argument("--MapFile",type=str,default="../KITTI-04/Map.yml")
    io_parser.add_argument("--KeyFrameIdKey",type=str,default="mnId")
    io_parser.add_argument("--FrameIdKey",type=str,default="mnFrameId")
    io_parser.add_argument("--KeyPointsKey",type=str,default="mvKeysUn")
    io_parser.add_argument("--MapPointKey",type=str,default="mvpMapPointsId")
    io_parser.add_argument("--CorrKey",type=str,default="mvpCorrKeyPointsId")
    io_parser.add_argument("--index_i",type=int,default=0)
    io_parser.add_argument("--index_j",type=int,default=1)
    io_parser.add_argument("--debug_log",type=str,default="")
    io_parser.add_argument("--extrinsic_file1",type=str,default="../KITTI-04/calib_res/he_rb_calib_04.txt")
    io_parser.add_argument("--extrinsic_file2",type=str,default="../KITTI-04/calib_res/iba_global_pl_04.txt")
    io_parser.add_argument("--Twc_file",type=str,default="../Twc.txt")
    io_parser.add_argument("--Twl_file",type=str,default="../Twl.txt")
    io_parser.add_argument("--save",type=str,default="../fig/04/cba_combine.png")
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--fps_sample",type=int,default=50)
    arg_parser.add_argument("--pixel_corr_dist",type=float,default=1.5)
    arg_parser.add_argument("--max_2d_nn",type=int, default=10)
    args = parser.parse_args()
    args.seq_id = "%02d"%args.seq
    return args


def get_fs_info(fs, keypt_nodename, mappt_nodename, corrkpt_nodename):
    kptnode = fs.getNode(keypt_nodename)
    keypts = [[int(kptnode.at(i).at(0).real()), int(kptnode.at(i).at(1).real())] for i in range(kptnode.size())]
    mapptnode = fs.getNode(mappt_nodename)
    mappt_indices = [int(mapptnode.at(i).real()) for i in range(mapptnode.size())]
    corrkpt_node = fs.getNode(corrkpt_nodename)
    corr_keypt_indices = [int(corrkpt_node.at(i).real()) for i in range(corrkpt_node.size())]
    return np.array(keypts), mappt_indices, corr_keypt_indices

def getMatchedId(src_mappt_indices:list, tgt_mappt_indices:list, src_corrkpt_indices:list, tgt_corrkpt_indices:list):
    src_matched_indices = []
    tgt_matched_indices = []
    for i in range(len(src_mappt_indices)):
        src_mappt_id = src_mappt_indices[i]
        if src_mappt_id in tgt_mappt_indices:
            tgt_id = tgt_mappt_indices.index(src_mappt_id)
            src_matched_indices.append(src_corrkpt_indices[i])
            tgt_matched_indices.append(tgt_corrkpt_indices[tgt_id])
    return np.array(src_matched_indices), np.array(tgt_matched_indices)  # Keypoint Indices


if __name__ == "__main__":
    np.random.seed(10)  # color consistency
    args = options()
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    extran1 = np.eye(4)
    extran_raw1 = np.loadtxt(args.extrinsic_file1)[:12]
    extran1[:3,:] = extran_raw1.reshape(3,4)
    extran2 = np.eye(4)
    extran_raw2 = np.loadtxt(args.extrinsic_file2)[:12]
    extran2[:3,:] = extran_raw2.reshape(3,4)
    KeyFramesFiles = list(sorted(os.listdir(args.KeyFrameDir)))
    KeyFramesFiles = [file for file in KeyFramesFiles if os.path.splitext(file)[1] == '.yml']
    FrameIdCfg = yaml.load(open(args.FrameIdFile,'r'), yaml.SafeLoader)
    mnIds:list = FrameIdCfg[args.KeyFrameIdKey]
    mFrameIds:list = FrameIdCfg[args.FrameIdKey]
    assert(args.index_i in mnIds and args.index_j in mnIds), "{} and {} must be in mnIds".format(args.index_i, args.index_j)
    src_kffile_index = mnIds.index(args.index_i)
    tgt_kffile_index = mnIds.index(args.index_j)
    src_file_index = mFrameIds[src_kffile_index]
    tgt_file_index = mFrameIds[tgt_kffile_index]
    src_KeyFrameFile = os.path.join(args.KeyFrameDir, KeyFramesFiles[src_kffile_index])
    tgt_KeyFrameFile = os.path.join(args.KeyFrameDir, KeyFramesFiles[tgt_kffile_index])
    src_fs = cv2.FileStorage(src_KeyFrameFile, cv2.FILE_STORAGE_READ)
    tgt_fs = cv2.FileStorage(tgt_KeyFrameFile, cv2.FILE_STORAGE_READ)
    src_keypts, src_mappt_indices, src_corrkpt_indices = get_fs_info(src_fs, args.KeyPointsKey, args.MapPointKey, args.CorrKey)
    tgt_keypts, tgt_mappt_indices, tgt_corrkpt_indices = get_fs_info(tgt_fs, args.KeyPointsKey, args.MapPointKey, args.CorrKey)
    intran = calibStruct.K_cam0
    src_matched_pts_idx, tgt_matched_pts_idx = getMatchedId(src_mappt_indices, tgt_mappt_indices, src_corrkpt_indices, tgt_corrkpt_indices)
    src_pcd_arr = dataStruct.get_velo(src_file_index)[:,:3]  # [N, 3]
    tgt_pcd_arr = dataStruct.get_velo(tgt_file_index)[:,:3]  # [N, 3]
    src_img = np.array(dataStruct.get_cam0(src_file_index))  # [H, W, 3]
    tgt_img = np.array(dataStruct.get_cam0(tgt_file_index))  # [H, W, 3]
    img_shape = src_img.shape[:2]
    src_pose = dataStruct.poses[src_file_index]
    tgt_pose = dataStruct.poses[tgt_file_index]
    camera_motion:np.ndarray = inv_pose(tgt_pose) @ src_pose  # Tc2w * Twc1
    src_matched_pts_raw = src_keypts[src_matched_pts_idx]
    tgt_matched_pts_raw = tgt_keypts[tgt_matched_pts_idx]
    print("Keypoint-Keypoint Matched:{}".format(len(src_matched_pts_raw)))
    src_pcd_camcoord = nptran(src_pcd_arr, extran1)
    proj_src_pcd, src_rev = npproj(src_pcd_camcoord, np.eye(4), intran, src_img.shape)
    src_pcd_camcoord = src_pcd_camcoord[src_rev]
    tgt_pcd_camcoord = nptran(src_pcd_camcoord, camera_motion)
    proj_tgt_pcd, tgt_rev = npproj(tgt_pcd_camcoord, np.eye(4), intran, tgt_img.shape)
    tgt_pcd_camcoord = tgt_pcd_camcoord[tgt_rev]
    src_2d_kdtree = cKDTree(proj_src_pcd, leafsize=10)
    src_dist, src_pcd_query = src_2d_kdtree.query(src_matched_pts_raw, 1, eps=1e-4, p=2, workers=-1)
    src_dist_rev = src_dist <= args.pixel_corr_dist**2
    src_matched_pts = src_matched_pts_raw[src_dist_rev]
    tgt_matched_pts = tgt_matched_pts_raw[src_dist_rev]
    src_pcd_query = src_pcd_query[src_dist_rev]
    src_pcd_3d_query = np.arange(src_pcd_camcoord.shape[0])
    src_pcd_3d_query = src_pcd_3d_query[src_pcd_query]
    sp_pts3d = src_pcd_camcoord[src_pcd_3d_query]
    src_proj_pcd, _ = npproj(sp_pts3d, np.eye(4), intran, src_img.shape)
    tgt_pcd_arr = nptran(sp_pts3d, camera_motion)
    tgt_proj_pcd, tgt_proj_rev = npproj(tgt_pcd_arr, np.eye(4), intran, tgt_img.shape)
    src_matched_pts = src_matched_pts[tgt_proj_rev]
    tgt_matched_pts = tgt_matched_pts[tgt_proj_rev]
    err1 = np.mean(np.sqrt(np.sum((tgt_matched_pts - tgt_proj_pcd)**2,axis=1)))
    Nmatch1 = src_matched_pts.shape[0]
    if args.fps_sample > 0:
        src_matched_pts, src_proj_pcd, fps_indices = fps_sample_corr_pts(src_matched_pts, src_proj_pcd, args.fps_sample, return_idx=True)
        tgt_matched_pts = tgt_matched_pts[fps_indices,:]
        tgt_proj_pcd = tgt_proj_pcd[fps_indices,:]
    draw_tgt_img1, _ = draw4corrpoints(tgt_img, src_img,
        *list(map(lambda arr: arr.astype(np.int32),[tgt_matched_pts, tgt_proj_pcd, src_matched_pts, src_proj_pcd])),
        5,8)
    
    src_pcd_camcoord = nptran(src_pcd_arr, extran2)
    proj_src_pcd, src_rev = npproj(src_pcd_camcoord, np.eye(4), intran, src_img.shape)
    src_pcd_camcoord = src_pcd_camcoord[src_rev]
    tgt_pcd_camcoord = nptran(src_pcd_camcoord, camera_motion)
    proj_tgt_pcd, tgt_rev = npproj(tgt_pcd_camcoord, np.eye(4), intran, tgt_img.shape)
    tgt_pcd_camcoord = tgt_pcd_camcoord[tgt_rev]
    src_2d_kdtree = cKDTree(proj_src_pcd, leafsize=10)
    src_dist, src_pcd_query = src_2d_kdtree.query(src_matched_pts_raw, 1, eps=1e-4, p=2, workers=-1)
    src_dist_rev = src_dist <= args.pixel_corr_dist**2
    src_matched_pts = src_matched_pts_raw[src_dist_rev]
    tgt_matched_pts = tgt_matched_pts_raw[src_dist_rev]
    src_pcd_query = src_pcd_query[src_dist_rev]
    src_pcd_3d_query = np.arange(src_pcd_camcoord.shape[0])
    src_pcd_3d_query = src_pcd_3d_query[src_pcd_query]
    sp_pts3d = src_pcd_camcoord[src_pcd_3d_query]
    src_proj_pcd, _ = npproj(sp_pts3d, np.eye(4), intran, src_img.shape)
    tgt_pcd_arr = nptran(sp_pts3d, camera_motion)
    tgt_proj_pcd, tgt_proj_rev = npproj(tgt_pcd_arr, np.eye(4), intran, tgt_img.shape)
    src_matched_pts = src_matched_pts[tgt_proj_rev]
    tgt_matched_pts = tgt_matched_pts[tgt_proj_rev]
    err2 = np.mean(np.sqrt(np.sum((tgt_matched_pts - tgt_proj_pcd)**2,axis=1)))
    Nmatch2 = src_matched_pts.shape[0]
    if args.fps_sample > 0:
        src_matched_pts, src_proj_pcd, fps_indices = fps_sample_corr_pts(src_matched_pts, src_proj_pcd, args.fps_sample, return_idx=True)
        tgt_matched_pts = tgt_matched_pts[fps_indices,:]
        tgt_proj_pcd = tgt_proj_pcd[fps_indices,:]
    draw_tgt_img2, _ = draw4corrpoints(tgt_img, src_img,
        *list(map(lambda arr: arr.astype(np.int32),[tgt_matched_pts, tgt_proj_pcd, src_matched_pts, src_proj_pcd])),
        5,8)
    
    plt.figure(dpi=200)
    plt.subplot(2,1,1)
    plt.title("Frame {} - {} | Matched:{} | Mean Error:{:0.2f}".format(src_file_index+1, tgt_file_index+1, Nmatch1,err1))
    plt.imshow(draw_tgt_img1)
    plt.axis([0,img_shape[1],img_shape[0],0])
    plt.axis('off')
    plt.subplot(2,1,2)
    plt.title("Frame {} - {} | Matched:{} | Mean Error:{:0.2f}".format(src_file_index+1, tgt_file_index+1, Nmatch2,err2))
    plt.imshow(draw_tgt_img2)
    plt.axis([0,img_shape[1],img_shape[0],0])
    plt.axis('off')
    plt.tight_layout(pad=0.5,h_pad=0,w_pad=0)
    plt.savefig(args.save)

    