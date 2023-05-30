import pykitti
import numpy as np
import argparse
import os
import cv2
import open3d as o3d
import copy
# import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from .cv_tools import *

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
    kitti_parser.add_argument("--index_i",type=int,default=105)
    kitti_parser.add_argument("--index_j",type=int,default=108)
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--Twc_file",type=str,default="../Twc.txt")
    io_parser.add_argument("--Twl_file",type=str,default="../Twl.txt")
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--tsl_perturb",type=float,nargs=3,default=[0.1,-0.15,0.1])
    arg_parser.add_argument("--rot_perturb",type=float,nargs=3,default=[0,0,0])
    arg_parser.add_argument("--fps_sample",type=int,default=100)
    arg_parser.add_argument("--epipolar_threshold",type=float,default=2)
    arg_parser.add_argument("--icp_radius",type=float,default=0.1)
    arg_parser.add_argument("--corr_threshold",type=float,default=0.05)
    arg_parser.add_argument("--view",type=str2bool,default=True)
    args = parser.parse_args()
    args.seq_id = "%02d"%args.seq
    return args

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def drawlines(img1,img2,lines,pts1,pts2):
    ''' Copied from: https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html\n
    img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def drawcorrpoints(img1,img2,pts1,pts2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def EpipolarwithF(src_proj_pts:np.ndarray, tgt_proj_pts:np.ndarray, F:np.ndarray, threshold=2, dual=True, draw=True):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(tgt_proj_pts.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    if dual:
        lines2 = cv2.computeCorrespondEpilines(src_proj_pts.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        # img3, img2 = drawlines(src_img,tgt_img,lines2,src_proj_pts,tgt_proj_pts)
        # img0, img1 = drawcorrpoints(src_img, tgt_img, src_proj_pts, tgt_proj_pts)
        src_dist:np.ndarray = np.abs(np.sum(src_proj_pts * lines1[:,:2], axis=1)+lines1[:,-1])
        tgt_dist:np.ndarray = np.abs(np.sum(tgt_proj_pts * lines2[:,:2], axis=1)+lines2[:,-1])
        rev_mask = np.logical_and(src_dist < threshold, tgt_dist < threshold)
        d_mean = 0.5*(src_dist.mean() + tgt_dist.mean())
    else:
        src_dist:np.ndarray = np.abs(np.sum(src_proj_pts * lines1[:,:2], axis=1)+lines1[:,-1])
        rev_mask = src_dist < threshold
        d_mean = src_dist.mean()
    if draw:
        img5, img4 = drawlines(src_img,tgt_img,lines1,src_proj_pts,tgt_proj_pts)
        return img5, img4, rev_mask.sum(), d_mean
    else:
        return rev_mask.sum(), d_mean

def EpipolarwithoutF(src_proj_pts:np.ndarray, tgt_proj_pts:np.ndarray, threshold=2, dual=True, draw=True):
    F, mask = cv2.findFundamentalMat(src_proj_pts, tgt_proj_pts, cv2.FM_LMEDS)
    print("Fundamental Matrix:\n{}".format(F))
    # select Inliers
    src_proj_pts = src_proj_pts[mask.ravel()==1]
    tgt_proj_pts = tgt_proj_pts[mask.ravel()==1]
    if draw:
        img5, img4, Ncorr, d_mean = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold, dual, True)
        return img5, img4, Ncorr, F, d_mean
    else:
        Ncorr, d_mean = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold, dual, False)
        return Ncorr, F, d_mean



if __name__ == "__main__":
    args = options()
    augT = toMat(args.rot_perturb, args.tsl_perturb)
    print("augT:\n{}".format(augT))
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    extran = calibStruct.T_cam0_velo  # [4,4]
    print("GT TCL:\n{}".format(extran))
    print("GT TCL Rvec:{}\ntvec:{}".format(*toVec(extran)))
    aug_extran = augT @ extran
    intran = calibStruct.K_cam0
    camera_motion:np.ndarray = dataStruct.poses[args.index_j] @ inv_pose(dataStruct.poses[args.index_i])
    print("Motion:\n{}".format(camera_motion))
    F = computeF(camera_motion, intran)
    print("Motion Rvec:{}\ntvec:{}".format(*toVec(camera_motion)))
    src_pcd_arr = dataStruct.get_velo(args.index_i)[:,:3]  # [N, 3]
    tgt_pcd_arr = dataStruct.get_velo(args.index_j)[:,:3]  # [N, 3]
    src_img = np.array(dataStruct.get_cam0(args.index_i))  # [H, W, 3]
    tgt_img = np.array(dataStruct.get_cam0(args.index_j))  # [H, W, 3]
    img_shape = src_img.shape[:2]
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pcd_arr)
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pcd_arr)
    src_pcd.transform(inv_pose(extran) @ dataStruct.poses[args.index_i] @ extran)
    tgt_pcd.transform(inv_pose(extran) @ dataStruct.poses[args.index_j] @ extran)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, args.icp_radius, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p.transformation)
    src_pcd.transform(reg_p2p.transformation)
    # check correspondences
    corr_set = np.array(reg_p2p.correspondence_set)
    print("Raw Correspondences:{}".format(corr_set.shape[0]))
    src_pcd_corr_transformed = np.array(src_pcd.points)[corr_set[:,0]]
    tgt_pcd_corr_transformed = np.array(tgt_pcd.points)[corr_set[:,1]]
    corr_rev = np.sum((src_pcd_corr_transformed-tgt_pcd_corr_transformed)**2,axis=1) < args.corr_threshold ** 2
    corr_set = corr_set[corr_rev]
    # project LiDAR points onto the images
    src_pcd_corr = src_pcd_arr[corr_set[:,0]]
    tgt_pcd_corr = tgt_pcd_arr[corr_set[:,1]]
    print("Selected Correspondences:{}".format(src_pcd_corr.shape[0]))
    src_proj_pts, tgt_proj_pts = project_corr_pts(src_pcd_corr, tgt_pcd_corr, extran, intran, img_shape)
    print("Projected Points:{}".format(src_proj_pts.shape[0]))
    if(args.fps_sample > 0):
        src_proj_pts_sampled, tgt_proj_pts_sampled = fps_sample_corr_pts(src_proj_pts, tgt_proj_pts, args.fps_sample)  # can be replaced by farthest_point_down_sample of Open3D
        Ncorr, d_mean = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=False)
        proj2src, proj2tgt, *_ = EpipolarwithF(src_proj_pts_sampled, tgt_proj_pts_sampled, F, threshold=args.epipolar_threshold, draw=True)
    else:
        proj2src, proj2tgt, Ncorr, d_mean = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=True)
    # visualization
    print("Ground-truth Matches:{}, d-mean:{}".format(Ncorr,d_mean))
    plt.figure(figsize=[12,3.5],dpi=200)
    plt.subplot(2,2,1)
    plt.title("Total:{} | Matched:{}".format(src_proj_pts.shape[0],Ncorr))
    plt.imshow(proj2src)
    plt.subplot(2,2,2)
    plt.title("Mean Dist:{:0.6f}".format(d_mean))
    plt.imshow(proj2tgt)
    src_proj_pts, tgt_proj_pts = project_corr_pts(src_pcd_corr, tgt_pcd_corr, aug_extran, intran, img_shape)
    if(args.fps_sample > 0):
        src_proj_pts_sampled, tgt_proj_pts_sampled = fps_sample_corr_pts(src_proj_pts, tgt_proj_pts, args.fps_sample)
        augNcorr, aug_dmean = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=False)
        augproj2src, augproj2tgt, *_ = EpipolarwithF(src_proj_pts_sampled, tgt_proj_pts_sampled, F, threshold=args.epipolar_threshold, draw=True)
    else:
        augproj2src, augproj2tgt, augNcorr, aug_dmean = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=True)
    print("Augmented Matches:{}, d-mean:{}".format(augNcorr,aug_dmean))
    plt.subplot(2,2,3)
    plt.title("Total:{} | Matched:{}".format(src_proj_pts.shape[0],augNcorr))
    plt.imshow(augproj2src)
    plt.subplot(2,2,4)
    plt.title("Mean Dist:{:0.6f}".format(aug_dmean))
    plt.imshow(augproj2tgt)
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    # plt.savefig("../demo/ep_{:06d}_{:06d}.pdf".format(args.index_i, args.index_j))
    if args.view:
        draw_registration_result(src_pcd, tgt_pcd, np.eye(4))
    

    