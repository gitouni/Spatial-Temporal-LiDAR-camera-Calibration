import numpy as np
from scipy.spatial.transform import Rotation
from fps_v1 import FPS

def skew(x:np.ndarray):
    return np.array([[0,-x[2],x[1]],
                     [x[2],0,-x[0]],
                     [-x[1],x[0],0]])
    
def computeF(motion:np.ndarray, intran:np.ndarray):
    rotation = motion[:3,:3]
    translation_skew = skew(motion[:3,3])
    inv_intran:np.ndarray = np.linalg.inv(intran)
    return inv_intran.T @ translation_skew @ rotation @ inv_intran
    
def toVec(SE3:np.ndarray):
    """SE3 Matrix to rvec and tvec

    Args:
        SE3 (np.ndarray): 4x4 `np.ndarray`

    Returns:
        rvec, tvec: `np.ndarray`
    """
    R = Rotation.from_matrix(SE3[:3,:3])
    tvec = SE3[:3,3]
    return R.as_rotvec(), tvec

def toMat(rvec:np.ndarray, tvec:np.ndarray):
    """rvec and tvec to SE3 Matrix

    Args:
        rvec (`np.ndarray`): 1x3 rotation vector\n
        tvec (`np.ndarray`): 1x3 translation vector
        
    Returns:
        SE3: 4x4 `np.ndarray`
    """
    R = Rotation.from_rotvec(rvec)
    mat = np.eye(4)
    mat[:3,:3] = R.as_matrix()
    mat[:3,3] = tvec
    return mat
    

def readKITTIPCD(filename:str) -> list:
    data:np.ndarray = np.fromfile(filename, dtype=np.float32)
    return data.reshape(-1, 4)

def read_pose_file(filename:str) -> list:
    raw_data = np.loadtxt(filename)
    pose_list = []
    for line in raw_data:
        pose = np.eye(4)
        pose[:3, :] = line.reshape(3,4)
        pose_list.append(pose)
    return pose_list

def inv_pose(pose:np.ndarray):
    inv_pose = np.eye(4)
    inv_pose[:3,:3] = pose[:3,:3].T
    inv_pose[:3,3] = -inv_pose[:3,:3] @ pose[:3, 3]
    return inv_pose

def pose2motionList(pose_list:list) -> list:
    motion_list = []
    for i in range(len(pose_list)-1):
        motion = pose_list[i+1] @ inv_pose(pose_list[i])
        motion_list.append(motion)
    return motion_list

def pose2motion(src_pose:np.ndarray, tgt_pose:np.ndarray) -> list:
    return tgt_pose @ inv_pose(src_pose)

def nptran(pcd:np.ndarray, rigdtran:np.ndarray) -> np.ndarray:
    pcd_ = pcd.copy().T  # (N, 3) -> (3, N)
    pcd_ = rigdtran[:3, :3] @ pcd_ + rigdtran[:3, [3]]
    return pcd_.T

def npproj(pcd:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple):
    """_summary_

    Args:
        pcd (np.ndarray): Nx3\\
        extran (np.ndarray): 4x4\\
        intran (np.ndarray): 3x3\\
        img_shape (tuple): HxW\\

    Returns:
        _type_: uv (N,2), rev (N,)
    """
    H, W = img_shape
    pcd_ = nptran(pcd, extran)  # (N, 3)
    pcd_ = intran @ pcd_.T  # (3, N)
    u, v, w = pcd_[0], pcd_[1], pcd_[2]
    raw_index = np.arange(u.size)
    rev = w > 0
    raw_index = raw_index[rev]
    u = u[rev]/w[rev]
    v = v[rev]/w[rev]
    rev2 = (0<=u) * (u<W) * (0<=v) * (v<H)
    return np.stack((u[rev2],v[rev2]),axis=1), raw_index[rev2]  # (N, 2), (N,)

def project_corr_pts(src_pcd_corr:np.ndarray, tgt_pcd_corr:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple):
    src_proj_pts, src_rev_idx = npproj(src_pcd_corr, extran, intran, img_shape)
    tgt_proj_pts, tgt_rev_idx = npproj(tgt_pcd_corr, extran, intran, img_shape)
    src_in_tgt_idx = np.isin(src_rev_idx, tgt_rev_idx)
    tgt_in_src_idx = np.isin(tgt_rev_idx, src_rev_idx)
    src_proj_pts = src_proj_pts[src_in_tgt_idx]
    tgt_proj_pts = tgt_proj_pts[tgt_in_src_idx]
    src_proj_pts = src_proj_pts.astype(np.int32)
    tgt_proj_pts = tgt_proj_pts.astype(np.int32)
    return src_proj_pts, tgt_proj_pts

def project_corr_pts_idx(src_pcd_corr:np.ndarray, tgt_pcd_corr:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple):
    src_proj_pts, src_rev_idx = npproj(src_pcd_corr, extran, intran, img_shape)
    tgt_proj_pts, tgt_rev_idx = npproj(tgt_pcd_corr, extran, intran, img_shape)
    src_in_tgt_idx = np.isin(src_rev_idx, tgt_rev_idx)
    tgt_in_src_idx = np.isin(tgt_rev_idx, src_rev_idx)
    src_proj_pts = src_proj_pts[src_in_tgt_idx]
    tgt_proj_pts = tgt_proj_pts[tgt_in_src_idx]
    src_proj_pts = src_proj_pts.astype(np.int32)
    tgt_proj_pts = tgt_proj_pts.astype(np.int32)
    rev_idx = np.arange(src_rev_idx.size)
    return src_proj_pts, tgt_proj_pts, rev_idx[src_in_tgt_idx]


def fps_sample_corr_pts(src_pts:np.ndarray, tgt_pts:np.ndarray, n_sample=100, return_idx=False):
    assert(src_pts.shape[0] > 0), "Empty FPS Input: {} points.".format(src_pts.shape[0])
    fps = FPS(src_pts, n_sample)
    _, idx = fps.fit(return_index=True)
    src_pts_ = src_pts[idx]
    tgt_pts_ = tgt_pts[idx]
    if not return_idx:
        return src_pts_, tgt_pts_
    else:
        return src_pts_, tgt_pts_, idx