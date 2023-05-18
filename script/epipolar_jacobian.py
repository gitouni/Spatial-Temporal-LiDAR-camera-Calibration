import os
os.chdir(os.path.dirname(__file__))

import pykitti
import numpy as np
import argparse
import cv2
import open3d as o3d
import copy
# import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from tools import *
from scipy.optimize import minimize


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
    kitti_parser.add_argument("--index_j",type=int,default=106)
    
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--Twc_file",type=str,default="../Twc.txt")
    io_parser.add_argument("--Twl_file",type=str,default="../Twl.txt")
    
    arg_parser = parser.add_argument_group()
    arg_parser.add_argument("--tsl_perturb",type=float,nargs=3,default=[-0.1,0.15,0.1])
    arg_parser.add_argument("--rot_perturb",type=float,nargs=3,default=[0,0,0])
    arg_parser.add_argument("--fps_sample",type=int,default=100)
    arg_parser.add_argument("--epipolar_threshold",type=float,default=2)
    arg_parser.add_argument("--icp_radius",type=float,default=0.1)
    arg_parser.add_argument("--corr_threshold",type=float,default=0.05)
    arg_parser.add_argument("--view",type=str2bool,default=False)
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
        d_rmse = 0.5*(np.sqrt(np.sum(src_dist**2)/src_dist.size)+np.sqrt(np.sum(tgt_dist**2)/tgt_dist.size))
    else:
        src_dist:np.ndarray = np.abs(np.sum(src_proj_pts * lines1[:,:2], axis=1)+lines1[:,-1])
        rev_mask = src_dist < threshold
        d_rmse = np.sqrt(np.sum(src_dist**2/src_dist.size))
    if draw:
        img5, img4 = drawlines(src_img,tgt_img,lines1,src_proj_pts,tgt_proj_pts)
        return img5, img4, rev_mask.sum(), d_rmse
    else:
        return rev_mask.sum(), d_rmse

def EpipolarwithoutF(src_proj_pts:np.ndarray, tgt_proj_pts:np.ndarray, threshold=2, dual=True, draw=True):
    F, mask = cv2.findFundamentalMat(src_proj_pts, tgt_proj_pts, cv2.FM_LMEDS)
    print("Fundamental Matrix:\n{}".format(F))
    # select Inliers
    src_proj_pts = src_proj_pts[mask.ravel()==1]
    tgt_proj_pts = tgt_proj_pts[mask.ravel()==1]
    if draw:
        img5, img4, Ncorr, d_rmse = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold, dual, True)
        return img5, img4, Ncorr, F, d_rmse
    else:
        Ncorr, d_rmse = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold, dual, False)
        return Ncorr, F, d_rmse


def npproj_jac(pcd:np.ndarray, rvec:np.ndarray, tvec:np.ndarray, intran:np.ndarray, img_shape:tuple):
    distCoeffs = np.zeros(4)
    extran = toMat(rvec, tvec)
    _, rev_idx = npproj(pcd, extran, intran, img_shape)
    assert(pcd[rev_idx].shape[0] > 0)
    img_pts, jac = cv2.projectPoints(pcd[rev_idx], rvec, tvec, intran, distCoeffs)
    # img_pts, jac = cv2.projectPoints(pcd, rvec, tvec, intran, distCoeffs)
    jac:np.ndarray = jac.reshape(-1,2,jac.shape[-1])  # N, 2, 10+numDistCoeffs
    img_pts:np.ndarray = img_pts.reshape(-1,2)
    # rev_idx = np.arange(img_pts.shape[0])
    return img_pts, jac[...,:6], rev_idx

def project_corr_pts_jac(src_pcd_corr:np.ndarray, tgt_pcd_corr:np.ndarray, rvec:np.ndarray, tvec:np.ndarray, intran:np.ndarray, img_shape:tuple):
    src_proj_pts, src_proj_jac, src_rev_idx = npproj_jac(src_pcd_corr, rvec, tvec, intran, img_shape)
    tgt_proj_pts, tgt_proj_jac, tgt_rev_idx = npproj_jac(tgt_pcd_corr, rvec, tvec, intran, img_shape)
    src_in_tgt_idx = np.isin(src_rev_idx, tgt_rev_idx)
    tgt_in_src_idx = np.isin(tgt_rev_idx, src_rev_idx)
    src_proj_pts = src_proj_pts[src_in_tgt_idx]
    tgt_proj_pts = tgt_proj_pts[tgt_in_src_idx]
    src_proj_jac = src_proj_jac[src_in_tgt_idx]
    tgt_proj_jac = tgt_proj_jac[tgt_in_src_idx]
    src_proj_pts = src_proj_pts.astype(np.int32)
    tgt_proj_pts = tgt_proj_pts.astype(np.int32)
    return src_proj_pts, tgt_proj_pts, src_proj_jac, tgt_proj_jac

def epipolar_err_jac(src_proj_pts:np.ndarray, tgt_proj_pts:np.ndarray, F:np.ndarray):
    """Compute error and jacobian of Epipolar Constraints

    Args:
        src_proj_pts (np.ndarray): Nx2\n
        tgt_proj_pts (np.ndarray): Nx2\n
        F (np.ndarray): 3x3 Fundamental Matrix\n
        src_proj_jac (np.ndarray): Nx2x6 Jacobian Matrix of `src_proj_pts`\n
        tgt_proj_jac (np.ndarray): Nx2x6 Jacobian Matrix of `tgt_proj_pts`\n
    """
    def DABCxy(A:np.ndarray,B:np.ndarray,C:np.ndarray,x:np.ndarray,y:np.ndarray):
        N = A.size
        jac = np.zeros([N,5])
        D = A*x+B*y+C
        D2 = D**2
        den = A**2+B**2
        den2 = den**2
        jac[:,0] = (2*A*D2 - 2*D*x*A**2)/den2  # dF/dA
        jac[:,1] = (2*B*D2 - 2*D*y*B**2)/den2  # dF/dB
        jac[:,2] = 2*D/den  # dF/dC
        jac[:,2] = A*jac[:,2]  # dF/dx
        jac[:,3] = B*jac[:,2] # dF/dy
        return jac  # (N,5)
    
    assert(src_proj_pts.shape[0] == tgt_proj_pts.shape[0])
    FT = F.T
    lines1 = FT[:,:2] @ tgt_proj_pts.T + FT[:,[2]]  # (3,N)
    Df_DABCxy1 = DABCxy(lines1[0,:], lines1[1,:], lines1[2,:], src_proj_pts[:,0], src_proj_pts[:,1])  # (N,5)
    lines2 = F[:,:2] @ src_proj_pts.T + F[:,[2]]  # (3,N)
    Df_DABCxy2 = DABCxy(lines2[0,:], lines2[1,:], lines2[2,:], tgt_proj_pts[:,0], tgt_proj_pts[:,1])  # (N,5)
    DABCxy1_Dpts = np.zeros([5,4])
    DABCxy1_Dpts[:3,2:] = FT[:,:2]  # (3,2)  to target_proj_pts
    DABCxy1_Dpts[3:,:2] = np.eye(2)  # (2,2) to src_proj_pts
    DABCxy2_Dpts = np.zeros([5,4])
    DABCxy2_Dpts[:3,:2] = F[:,:2]  # (3,2) to src_proj_pts
    DABCxy2_Dpts[3:,2:] = np.eye(2)  # (2,2) to target_proj_pts
    err1 = (np.sum(lines1[:2,:]*src_proj_pts.T,axis=0) + lines1[2,:])**2/np.sum(lines1[:2,:]**2, axis=0)  # (Ax+By+C)^2/(A^2+B^2)
    err2 = (np.sum(lines2[:2,:]*tgt_proj_pts.T,axis=0) + lines2[2,:])**2/np.sum(lines2[:2,:]**2, axis=0)
    return err1 + err2, Df_DABCxy1 @ DABCxy1_Dpts + Df_DABCxy2 @ DABCxy2_Dpts  # err (N,) Jacobian (N,4)

class EpipolarOptimizer:
    def __init__(self, src_pcd_corr:np.ndarray, tgt_pcd_corr:np.ndarray, intran:np.ndarray, img_shape:tuple, F:np.ndarray):
        self.src_pcd = src_pcd_corr
        self.tgt_pcd = tgt_pcd_corr
        self.intran = intran
        self.img_shape = img_shape
        self.F = F
        self.input_buff = np.zeros(6)
        self.err_buf = None
        self.jac_buff = None
    
    def comput_err_jac(self, rvec:np.ndarray, tvec:np.ndarray):
        src_proj_pts, tgt_proj_pts, src_proj_jac, tgt_proj_jac = project_corr_pts_jac(self.src_pcd, self.tgt_pcd, rvec, tvec, self.intran, self.img_shape)
        proj_jac = np.concatenate((src_proj_jac, tgt_proj_jac), axis=1)  # (N,2,6), (N,2,6) -> (N,4,6)
        epipolar_err, epipolar_jac = epipolar_err_jac(src_proj_pts, tgt_proj_pts, self.F)  # (N,), (N,4) 
        jac = np.zeros(6)
        err = np.mean(epipolar_err)
        for i in range(jac.size):
            jac[i] = np.mean(epipolar_jac * proj_jac[...,i])
        self.jac_buff = jac
        return err, jac
    
    def compute_err(self, x:np.ndarray):
        if (self.err_buf is not None) and np.allclose(x, self.input_buff):
            return self.err_buf
        err, jac = self.comput_err_jac(x[:3], x[3:])
        self.input_buff = x.copy()
        self.err_buf = err
        self.jac_buff = jac
        return err
    
    def compute_jac(self,x:np.ndarray):
        if (self.jac_buff is not None) and np.allclose(x, self.input_buff):
            return self.jac_buff
        err, jac = self.comput_err_jac(x[:3], x[3:])
        self.input_buff = x.copy()
        self.err_buf = err
        self.jac_buff = jac
        return jac


if __name__ == "__main__":
    args = options()
    augT = toMat(args.rot_perturb, args.tsl_perturb)
    print("augT:\n{}".format(augT))
    dataStruct = pykitti.odometry(args.base_dir, args.seq_id)
    calibStruct = dataStruct.calib
    extran = calibStruct.T_cam0_velo  # [4,4]
    aug_extran = augT @ extran
    intran = calibStruct.K_cam0
    motion:np.ndarray = dataStruct.poses[args.index_j] @ inv_pose(dataStruct.poses[args.index_i])
    print("Motion:\n{}".format(motion))
    F = computeF(motion, intran)
    # F = F/np.linalg.norm(F,ord='fro')  # Normalize
    print("Fundamental Matrix:\n{}".format(F))
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
    src_pcd.transform(reg_p2p.transformation)
    checker = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(args.icp_radius)
    is_aligned = checker.Check(src_pcd, tgt_pcd, reg_p2p.correspondence_set,np.eye(4))
    print("Is Aligned:{}".format(is_aligned))
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
    optimizer = EpipolarOptimizer(src_pcd_corr, tgt_pcd_corr, intran, img_shape, F)
    rvec0, tvec0 = toVec(aug_extran)
    x0 = np.concatenate((rvec0,tvec0))
    print("Initial Error:{}".format(optimizer.compute_err(x0)))
    res = minimize(optimizer.compute_err, x0,  method='Nelder-Mead', jac=optimizer.compute_jac,
               options={'disp': 100})
    print("Final Error:{}".format(res.fun))
    predict_extran = toMat(res.x[:3], res.x[3:])
    print("Aug:\n{}".format(aug_extran))
    print("Predict:\n{}".format(predict_extran))
    print("GT:\n{}".format(extran))
    src_proj_pts, tgt_proj_pts = project_corr_pts(src_pcd_corr, tgt_pcd_corr, predict_extran, intran, img_shape)
    Predict_F, _ = cv2.findFundamentalMat(src_proj_pts, tgt_proj_pts, cv2.FM_LMEDS)
    print("Predict Fundamental Matrix:\n{}".format(Predict_F))
    print("Projected Points:{}".format(src_proj_pts.shape[0]))
    if(args.fps_sample > 0):
        src_proj_pts_sampled, tgt_proj_pts_sampled = fps_sample_corr_pts(src_proj_pts, tgt_proj_pts, args.fps_sample)  # can be replaced by farthest_point_down_sample of Open3D
        Ncorr, d_rmse = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=False)
        proj2src, proj2tgt, *_ = EpipolarwithF(src_proj_pts_sampled, tgt_proj_pts_sampled, F, threshold=args.epipolar_threshold, draw=True)
    else:
        proj2src, proj2tgt, Ncorr, d_rmse = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=True)
    # visualization
    print("Ground-truth Matches:{}, d-rmse:{}".format(Ncorr,d_rmse.item()))
    plt.figure(figsize=[12,3.5],dpi=200)
    plt.subplot(2,2,1)
    plt.title("Total:{} | Matched:{}".format(src_proj_pts.shape[0],Ncorr))
    plt.imshow(proj2src)
    plt.subplot(2,2,2)
    plt.title("RMSE Dist:{:0.6f}".format(d_rmse.item()))
    plt.imshow(proj2tgt)
    src_proj_pts, tgt_proj_pts = project_corr_pts(src_pcd_corr, tgt_pcd_corr, aug_extran, intran, img_shape)
    if(args.fps_sample > 0):
        src_proj_pts_sampled, tgt_proj_pts_sampled = fps_sample_corr_pts(src_proj_pts, tgt_proj_pts, args.fps_sample)
        augNcorr, aug_drmse = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=False)
        augproj2src, augproj2tgt, *_ = EpipolarwithF(src_proj_pts_sampled, tgt_proj_pts_sampled, F, threshold=args.epipolar_threshold, draw=True)
    else:
        augproj2src, augproj2tgt, augNcorr, aug_drmse = EpipolarwithF(src_proj_pts, tgt_proj_pts, F, threshold=args.epipolar_threshold, draw=True)
    print("Augmented Matches:{}, d-rmse:{}".format(augNcorr,aug_drmse))
    plt.subplot(2,2,3)
    plt.title("Total:{} | Matched:{}".format(src_proj_pts.shape[0],augNcorr))
    plt.imshow(augproj2src)
    plt.subplot(2,2,4)
    plt.title("RMSE Dist:{:0.6f}".format(aug_drmse))
    plt.imshow(augproj2tgt)
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    

    