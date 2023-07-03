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
from scipy.interpolate import BSpline, make_interp_spline
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
    kitti_parser.add_argument("--seq",type=int,default=0,choices=[i for i in range(11)])
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--KeyFrameDir",type=str,default="../KITTI-00/KeyFrames")
    io_parser.add_argument("--FrameIdFile",type=str,default="../KITTI-00/FrameId.yml")
    io_parser.add_argument("--KeyFrameIdKey",type=str,default="mnId")
    io_parser.add_argument("--FrameIdKey",type=str,default="mnFrameId")
    io_parser.add_argument("--KeyPointsKey",type=str,default="mvKeysUn")
    io_parser.add_argument("--MapPointKey",type=str,default="mvpMapPointsId")
    io_parser.add_argument("--CorrKey",type=str,default="mvpCorrKeyPointsId")
    io_parser.add_argument("--index_i",type=int,default=0)
    io_parser.add_argument("--index_j",type=int,default=3)
    io_parser.add_argument("--debug_log",type=str,default="")
    io_parser.add_argument("--Twc_file",type=str,default="../Twc.txt")
    io_parser.add_argument("--Twl_file",type=str,default="../Twl.txt")
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--fps_sample",type=int,default=-1)
    arg_parser.add_argument("--pixel_corr_dist",type=float,default=1.5)
    arg_parser.add_argument("--max_2d_nn",type=int, default=30)
    arg_parser.add_argument("--max_2d_std",type=float,default=0.05)
    arg_parser.add_argument("--max_depth_diff",type=float,default=0.1)
    arg_parser.add_argument("--gpr_radius",type=float,default=0.6)
    arg_parser.add_argument("--gpr_max_pts",type=int,default=30)
    arg_parser.add_argument("--gpr_max_cov",type=float,default=0.1)
    arg_parser.add_argument("--gpr",type=str2bool,default=True)
    arg_parser.add_argument("--gpr_opt",type=str2bool, default=True)
    arg_parser.add_argument("--mp",type=str2bool,default=True)
    arg_parser.add_argument("--depth_conflict",type=str2bool,default=False)
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

def getMatchedIdviaMap(mapnode, src_map_ptid:np.ndarray, tgt_map_ptid:np.ndarray, src_index:int, tgt_index:int):
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

def bspline_interp(train_X:np.ndarray, train_Y:np.ndarray, test_X:np.ndarray):
    bspline:BSpline = make_interp_spline(train_X, train_Y)
    if np.ndim(test_X) == 1:
        return bspline(test_X[None,:]).squeeze()
    else:
        return bspline(test_X)

def superpixel_approx(pcd:np.ndarray, query:np.ndarray, pixels:np.ndarray, intran:np.ndarray, radius:float, max_pts:int, img_shape:tuple):
    def gpr_approx(idx):
        query_idx = query[idx]
        pixel = pixels[idx]
        query_pt = pcd[query_idx]
        neigh_dist, neigh_indices = pcd_kdtree.query(query_pt, k = max_pts)
        neigh_rev = neigh_dist < radius
        neigh_indices = neigh_indices[neigh_rev]
        neigh_pts = pcd[neigh_indices, :]
        proj_pts, proj_rev = npproj(neigh_pts, np.eye(4), intran, img_shape)
        if proj_rev.sum() < 10:
            return query_pt
        neigh_pts = neigh_pts[proj_rev]
        depth = neigh_pts[:,-1]
        # try:
        gpr = GPR(optimize=args.gpr_opt)
        gpr.params['l'] = 5
        gpr.params['sigma_f'] = 0.1
        gpr.fit(proj_pts, 1/depth)  # use inverse depth
        # print_params(gpr.params)
        pz = gpr.predict(pixel[None,:]).item()
        pz = 1/pz
        px = (pixel[0] - cx) / fx * pz
        py = (pixel[1] - cy) / fy * pz
        return np.array([px, py, pz])
        # except Exception as e:
        #     return query_pt
    
    pcd_kdtree = cKDTree(pcd, leafsize=30)
    N = query.shape[0]
    assert(query.shape[0] == pixels.shape[0])
    fx = intran[0,0]
    fy = intran[1,1]
    cx = intran[0,2]
    cy = intran[1,2]
    if not args.mp:
        sp_pts = np.zeros([N,3])
        for i in range(N):
            sp_pts[i, :] = gpr_approx(i)
    else:
        parallel_res = Parallel(n_jobs=-1)(delayed(gpr_approx)(idx) for idx in range(query.shape[0]))
        sp_pts = np.array(parallel_res)
    return sp_pts


if __name__ == "__main__":
    np.random.seed(10)  # color consistency
    args = options()
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    extran = calibStruct.T_cam0_velo  # [4,4]
    print("GT TCL:\n{}".format(extran))
    print("GT TCL Rvec:{}\ntvec:{}".format(*toVec(extran)))
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
    src_pose = dataStruct.poses[src_file_index]
    tgt_pose = dataStruct.poses[tgt_file_index]
    camera_motion:np.ndarray = inv_pose(tgt_pose) @ src_pose  # Tc2w * Twc1
    src_matched_pts = src_keypts[src_matched_pts_idx]
    tgt_matched_pts = tgt_keypts[tgt_matched_pts_idx]
    print("Matched Keypoints:{}".format(len(src_matched_pts)))
    src_pcd_camcoord = nptran(src_pcd_arr, extran)
    proj_src_pcd, src_rev = npproj(src_pcd_camcoord, np.eye(4), intran, src_img.shape)
    src_pcd_camcoord = src_pcd_camcoord[src_rev]
    src_2d_kdtree = cKDTree(proj_src_pcd, leafsize=10)
    if args.depth_conflict:
        src_dists, src_pcd_querys = src_2d_kdtree.query(src_matched_pts, args.max_2d_nn, eps=1e-4, p=1, workers=-1)
        src_dist_rev = np.ones(src_dists.shape[0], dtype=np.bool8)
        src_pcd_query = np.zeros(src_dists.shape[0], dtype=np.int32)
        for i in range(src_matched_pts.shape[0]):
            dist = src_dists[i]
            dist_rev:np.ndarray = dist < args.pixel_corr_dist
            if dist_rev.sum() == 0:
                src_dist_rev[i] = False
                continue
            dist = dist[dist_rev]
            query = src_pcd_querys[i][dist_rev]
            if dist_rev.sum() == 1:
                src_pcd_query[i] = query[0]
                continue
            dist_std = np.std(dist)
            if dist_std < args.max_2d_std:
                src_pcd_query[i] = query[0]
            else:
                depths = src_pcd_camcoord[query,2]  # same permutation with dist
                depth_sortidx = np.argsort(depths)
                sorted_depths = depths[depth_sortidx]
                depth_diffs = np.diff(sorted_depths, n=1, prepend=sorted_depths[0])
                dist_rev2 = depth_diffs < args.max_depth_diff
                query_idx = np.min(depth_sortidx[dist_rev2])
                src_pcd_query[i] = query[query_idx]
                if query_idx != 0:
                    print("Before: {}, After: {}".format(query[0], query[query_idx]))
    else:
        src_dist, src_pcd_query = src_2d_kdtree.query(src_matched_pts, 1, eps=1e-4, p=1, workers=-1)
        src_dist_rev = src_dist < args.pixel_corr_dist
    src_matched_pts = src_matched_pts[src_dist_rev]
    tgt_matched_pts = tgt_matched_pts[src_dist_rev]
    src_pcd_query = src_pcd_query[src_dist_rev]
    src_pcd_3d_query = np.arange(src_pcd_camcoord.shape[0])
    src_pcd_3d_query = src_pcd_3d_query[src_pcd_query]
    if args.gpr:
        sp_pts3d = superpixel_approx(src_pcd_camcoord, src_pcd_3d_query, src_matched_pts, intran, args.gpr_radius, args.gpr_max_pts, src_img.shape)
    else:
        sp_pts3d = src_pcd_camcoord[src_pcd_3d_query]
    src_proj_pcd, _ = npproj(sp_pts3d, np.eye(4), intran, src_img.shape)
    tgt_pcd_arr = nptran(sp_pts3d, camera_motion)
    tgt_proj_pcd, tgt_proj_rev = npproj(tgt_pcd_arr, np.eye(4), intran, tgt_img.shape)
    src_matched_pts = src_matched_pts[tgt_proj_rev]
    tgt_matched_pts = tgt_matched_pts[tgt_proj_rev]
    err = np.mean(np.sqrt(np.sum((tgt_matched_pts - tgt_proj_pcd)**2,axis=1)))
    draw_tgt_img, draw_src_img = draw4corrpoints(tgt_img, src_img, *list(map(lambda arr: arr.astype(np.int32),[tgt_matched_pts, tgt_proj_pcd, src_matched_pts, src_proj_pcd])))
    
    plt.figure(dpi=200)
    plt.subplot(2,1,1)
    plt.title("Frame {} - {} | Matched:{} | error:{:0.4}".format(src_file_index+1, tgt_file_index+1, tgt_matched_pts.shape[0],err))
    plt.imshow(draw_src_img)
    plt.subplot(2,1,2)
    plt.imshow(draw_tgt_img)
    plt.show()

    