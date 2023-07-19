import numpy as np
import cv2

def canny(img:np.ndarray,T1:int=200,T2:int=50):
    if len(img.shape) == 3:
        img_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img_ = img.copy()
        # img_ = cv2.bilateralFilter(img_,9,50,150)
        # img_ = cv2.equalizeHist(img_)
        
    return cv2.Canny(img_,T1,T2)

def inverse_distance_transform(edge:np.ndarray,alpha:float,gamma:float):
    inv_edge = np.array(edge==0,dtype=np.uint8)  # edge: zero; non-edge: non-zero
    inv_map = cv2.distanceTransform(inv_edge,cv2.DIST_C,maskSize=5)  # max(|x-i|,|y-j|)
    D = alpha*edge + (1-alpha)*255*gamma**inv_map
    return np.uint8(D)


def cal_pitch(X:np.ndarray):
    R = np.sqrt(X[:,0]**2+X[:,1]**2)  # [n,]
    pitch = np.arctan(X[:,2]/R)  #[n,]
    return pitch

def cal_yaw(X:np.ndarray):
    x,y = X[:,0], X[:,1]
    yaw = np.arctan2(y,x)  # invert x & y
    return yaw

def range_dif(x:np.ndarray):
    x_df, x_bk = np.zeros_like(x), np.zeros_like(x)
    x_df[:-1] = np.abs(x[1:] - x[:-1])
    x_bk[1:] = np.abs(x[:-1] - x[1:])
    return np.maximum(x_df,x_bk)

def yaw_diff(line_scan:np.ndarray):
    x = cal_yaw(line_scan)
    x_df, x_bk = np.zeros_like(x), np.zeros_like(x)
    x_df[:-1] = np.abs(x[1:] - x[:-1])
    x_bk[1:] = np.abs(x[:-1] - x[1:])
    return np.maximum(x_df, x_bk) * 180 / np.pi


def sort_pcd(unsrt_pcd:np.ndarray,scan_num:int=16)->list:
    srt_pcd = list()
    pitch = np.degrees(cal_pitch(unsrt_pcd))
    scan_idx = np.array((pitch+15)/2+0.5,dtype=np.int32)  # VLP16 refer to A-LOAM
    for i in range(scan_num):
        srt_pcd.append(unsrt_pcd[scan_idx==i])
    return srt_pcd

def sort_pcd_with_idx(unsrt_pcd:np.ndarray,scan_num:int=16)->list:
    srt_pcd = list()
    srt_idx = list()
    pitch = np.degrees(cal_pitch(unsrt_pcd))
    idx = np.arange(len(pitch))
    scan_idx = np.array((pitch+15)/2+0.5,dtype=np.int32)  # VLP16 refer to A-LOAM
    for i in range(scan_num):
        mask = scan_idx==i
        srt_pcd.append(unsrt_pcd[mask])
        srt_idx.append(idx[mask])
    return srt_pcd, srt_idx

def pcd_dsc(pcd:np.ndarray,dis_thre=0.3, nan_extract=False, max_yaw_diff=0.5):
    srt_pcd = sort_pcd(pcd)
    dsc_pcd = np.empty((0,3),dtype=np.float32)
    for line_scan in srt_pcd:
        pcd_range = np.linalg.norm(line_scan,axis=1)  # (n,)
        dif_pcd_range = range_dif(pcd_range)
        if(nan_extract):
            dif_yaw_deg = yaw_diff(line_scan)
            rev = np.logical_or(dif_pcd_range > dis_thre, dif_yaw_deg > max_yaw_diff)
        else:
            rev = (dif_pcd_range > dis_thre)
        dsc_pcd = np.vstack((dsc_pcd,line_scan[rev]))
    return dsc_pcd

def pcd_dsc_with_idx(pcd:np.ndarray, dis_thre=0.3, nan_extract=False, max_yaw_diff=0.5):
    srt_pcd, srt_idx = sort_pcd_with_idx(pcd)
    dsc_pcd = np.empty((0,3),dtype=np.float32)
    dst_idx = np.empty((0,3),dtype=np.int32)
    for line_scan,idx_scan in zip(srt_pcd,srt_idx):
        pcd_range = np.linalg.norm(line_scan,axis=1)  # (n,)
        dif_pcd_range = range_dif(pcd_range)
        if(nan_extract):
            dif_yaw_deg = yaw_diff(pcd)
            rev = np.logical_or(dif_pcd_range > dis_thre, dif_yaw_deg > max_yaw_diff)
        else:
            rev = (dif_pcd_range > dis_thre)
        dsc_pcd = np.vstack((dsc_pcd,line_scan[rev]))
        dst_idx = np.vstack((dst_idx,idx_scan[rev]))
    return dsc_pcd, dst_idx


