import argparse
import numpy as np
from cv_tools import toVec
from problem import nptrans,binary_projection,CorrLoss
from edge import pcd_dsc
import pykitti as pyk
import os
from scipy.optimize import minimize
from tqdm import tqdm

os.chdir(os.path.abspath(os.path.dirname(__file__)))
def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--data_dir",type=str,default="/data/DATA/data_odometry/dataset/")
    io_parser.add_argument("--seq",type=int,default=0,choices=[0,2,3,4,5])
    io_parser.add_argument("--calib_dir",type=str,default="../KITTI-00/calib_res/")
    io_parser.add_argument("--init_calib",type=str,default="he_rb_calib_00.txt")
    io_parser.add_argument("--save_calib",type=str,default="edge_calib.csv")
    io_parser.add_argument("--data_skip",type=int,default=4)
    io_parser.add_argument("--pcd_skip",type=int,default=1)
    
    runtime_parser = parser.add_argument_group()
    runtime_parser.add_argument("--pcd_diff_threshold",type=float,default=0.3)
    runtime_parser.add_argument("--pcd_diff_yaw",type=float,default=0.5)
    runtime_parser.add_argument("--canny_max",type=int,default=120)
    runtime_parser.add_argument("--canny_min",type=int,default=20)
    runtime_parser.add_argument("--inv_dis_alpha",type=float,default=0.33)
    runtime_parser.add_argument("--inv_dis_gamma",type=float,default=0.98)
    runtime_parser.add_argument("--max_powell_iter",type=int,default=100)
    runtime_parser.add_argument("--delta_x",type=float,nargs=6,default=[0.15,0.15,0.15,0.3,0.3,0.3])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    init_calib = np.eye(4)
    init_calib_data = np.loadtxt(os.path.join(args.calib_dir, args.init_calib))[:12]
    init_calib[:3,:] = init_calib_data.reshape(3,4)
    pyk_data = pyk.odometry(args.data_dir,sequence="%02d"%args.seq)
    print("Sequence %02d has %d frames of data."%(args.seq, len(pyk_data)))
    intran = pyk_data.calib.K_cam0
    init_x0 = np.zeros(6)
    init_x0[:3], init_x0[3:] = toVec(init_calib)
    with open(os.path.join(args.calib_dir, args.save_calib),'w') as f:
        for i in tqdm(range(0,len(pyk_data),args.data_skip)):
            img = np.array(pyk_data.get_cam0(i))
            H, W = img.shape[:2]
            pcd:np.ndarray = pyk_data.get_velo(i)[::args.pcd_skip,:3]
            pcd = pcd_dsc(pcd, args.pcd_diff_threshold, True, args.pcd_diff_yaw)
            miscalib_pcd = nptrans(pcd.T, init_calib)
            u,v,mis_rev = binary_projection((H,W),intran,miscalib_pcd)
            Loss = CorrLoss(intran, (H,W), miscalib_pcd,
                img, args.canny_max, args.canny_min, args.inv_dis_alpha, args.inv_dis_gamma)
            bnds = [(x-dx,x+dx) for x,dx in zip(init_x0.tolist(),args.delta_x)]
            res = minimize(Loss.loss,init_x0,method="Nelder-Mead",bounds=bnds)
            f.write("{:0.4f} {:0.4f} {:0.4f} {:0.4f} {:0.4f} {:0.4f}\n".format(*res.x))
    
    