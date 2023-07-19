import numpy as np
import open3d as o3d
from edge import inverse_distance_transform, canny
from cv_tools import toMat

def voxel_downsample(pcd_arr:np.ndarray,voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr)
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.array(pcd.points)

def pcd_projection(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray,range:np.ndarray):
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    u = u[rev]
    v = v[rev]
    r = range[rev]
    return u,v,r,rev

def binary_projection(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray):
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    return u,v,rev

def nptrans(pcd:np.ndarray,G:np.ndarray)->np.ndarray:
    R,t = G[:3,:3], G[:3,[3]]  # (3,3), (3,1)
    return R @ pcd + t


class CorrLoss:
    def __init__(self,intran:np.ndarray,img_shape:tuple,raw_pcd:np.ndarray,raw_img,
                 canny_max:int=200, canny_min:int=20,
                 alpha:float=0.33,gamma:float=0.98):
        self.intran = intran  # (3,3)
        self.img_shape = img_shape  # (2,) (H,W)
        edge = canny(raw_img,canny_max,canny_min)
        self.inv_map = np.array(inverse_distance_transform(edge,alpha,gamma),dtype=np.float32)/255.  # (H,W) 0~1
        self.raw_pcd = raw_pcd
        
    def loss(self,x:np.ndarray):
        Tr:np.ndarray = toMat(x[:3],x[3:])
        pred_pcd = nptrans(self.raw_pcd,Tr)
        u,v,rev = binary_projection(self.img_shape,self.intran,pred_pcd)
        depth_img = np.zeros(self.img_shape)
        depth_img[v[rev],u[rev]] = 1
        return -(depth_img*self.inv_map).mean()
    
    def matloss(self,Tr:np.ndarray):
        pred_pcd = nptrans(self.raw_pcd,Tr)
        u,v,rev = binary_projection(self.img_shape,self.intran,pred_pcd)
        depth_img = np.zeros(self.img_shape)
        depth_img[v[rev],u[rev]] = 1
        return -(depth_img*self.inv_map).mean()
    