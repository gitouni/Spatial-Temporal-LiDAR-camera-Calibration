import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ax_label_fontsize",type=int,default=14)
    parser.add_argument("--ax_legend_fontsize",type=int,default=14)
    parser.add_argument("--ax_marker_size",type=int,default=10)
    parser.add_argument("--base_sim3",type=str,default="../KITTI-04/calib_res/gt_calib_04.txt")
    parser.add_argument("--rot_step",type=float,default=1)
    parser.add_argument("--rot_range",type=float,default=10)
    parser.add_argument("--tsl_step",type=float,default=0.01)
    parser.add_argument("--tsl_range",type=float,default=0.1)
    parser.add_argument("--scale_step",type=float,default=0.1)
    parser.add_argument("--scale_range",type=float,default=1)
    parser.add_argument("--he_threshold",type=float,default=0.03)
    parser.add_argument("--res_sim3",type=str,default="../demo/sim3_res_04.txt")
    parser.add_argument("--res_dir",type=str,default="../fig/")
    parser.add_argument("--fig_size",type=float,nargs=2,default=[8,3])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    parameters = {"xtick.labelsize":args.ax_label_fontsize,
                "ytick.labelsize":args.ax_label_fontsize,
                "font.size":args.ax_label_fontsize,
                "legend.fontsize":args.ax_legend_fontsize,
                "lines.markersize":args.ax_marker_size}
    plt.rcParams.update(parameters)
    rot_offset = np.arange(-args.rot_range, args.rot_range+1e-8, args.rot_step)  # right bound included
    tsl_offset = np.arange(-args.tsl_range, args.tsl_range+1e-8, args.tsl_step)*100  # right bound included m -> cm
    scale_offset = np.arange(-args.scale_range, args.scale_range+1e-8, args.scale_step)  # right bound included
    Nrot = rot_offset.size
    Ntsl = tsl_offset.size
    Nscale = scale_offset.size
    res_sim3 = np.loadtxt(args.res_sim3)
    # roll
    fig = plt.figure(figsize=args.fig_size)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data = res_sim3[:Nrot,:]
    f1 = data[:,0]
    f2 = data[:,1]
    C = data[:,2]
    nrev = C > args.he_threshold
    x = rot_offset
    plot_f1 = ax.plot(x,f1,'b-',marker='.')
    plot_f2 = ax2.plot(x,f2,'r-',marker='.')
    plot_C = ax.plot(x[nrev],f1[nrev],'k^',markersize=7)
    ax2.plot(x[nrev],f2[nrev],'k^',markersize=7)
    # ax.set_xlabel("roll offset ($^\circ$)")
    ax.set_ylabel("F1")
    ax2.set_ylabel("F2")
    plt.legend(plot_f1+plot_f2+plot_C,['F1','F2','IFeas'],loc='upper center',frameon=True,ncol=2)
    ax.grid(True)
    # plt.tight_layout()
    plt.savefig(os.path.join(args.res_dir,"cba_roll_offset.pdf"))
    # pitch
    fig = plt.figure(figsize=args.fig_size)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data = res_sim3[Nrot:2*Nrot,:]
    f1 = data[:,0]
    f2 = data[:,1]
    C = data[:,2]
    nrev = C > args.he_threshold
    x = rot_offset
    plot_f1 = ax.plot(x,f1,'b-',marker='.')
    plot_f2 = ax2.plot(x,f2,'r-',marker='.')
    plot_C = ax.plot(x[nrev],f1[nrev],'k^',markersize=7)
    ax2.plot(x[nrev],f2[nrev],'k^',markersize=7)
    plt.legend(plot_f1+plot_f2+plot_C,['F1','F2','IFeas'],loc='upper center',frameon=True,ncol=2)
    ax.grid(True)
    # ax.set_xlabel("pitch offset ($^\circ$)")
    ax.set_ylabel("F1")
    ax2.set_ylabel("F2")
    # plt.tight_layout()
    plt.savefig(os.path.join(args.res_dir,"cba_pitch_offset.pdf"))
    # yaw
    fig = plt.figure(figsize=args.fig_size)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data = res_sim3[Nrot*2:Nrot*3,:]
    f1 = data[:,0]
    f2 = data[:,1]
    C = data[:,2]
    nrev = C > args.he_threshold
    x = rot_offset
    plot_f1 = ax.plot(x,f1,'b-',marker='.')
    plot_f2 = ax2.plot(x,f2,'r-',marker='.')
    plot_C = ax.plot(x[nrev],f1[nrev],'k^',markersize=7)
    ax2.plot(x[nrev],f2[nrev],'k^',markersize=7)
    plt.legend(plot_f1+plot_f2+plot_C,['F1','F2','IFeas'],loc='upper center',frameon=True,ncol=2)
    ax.grid(True)
    # ax.set_xlabel("yaw offset ($^\circ$)")
    ax.set_ylabel("F1")
    ax2.set_ylabel("F2")
    # plt.tight_layout()
    plt.savefig(os.path.join(args.res_dir,"cba_yaw_offset.pdf"))
    # X
    fig = plt.figure(figsize=args.fig_size)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data = res_sim3[Nrot*3:Nrot*3+Ntsl,:]
    f1 = data[:,0]
    f2 = data[:,1]
    C = data[:,2]
    nrev = C > args.he_threshold
    x = tsl_offset
    plot_f1 = ax.plot(x,f1,'b-',marker='.')
    plot_f2 = ax2.plot(x,f2,'r-',marker='.')
    plot_C = ax.plot(x[nrev],f1[nrev],'k^',markersize=7)
    ax2.plot(x[nrev],f2[nrev],'k^',markersize=7)
    plt.legend(plot_f1+plot_f2+plot_C,['F1','F2','IFeas'],loc='upper center',frameon=True,ncol=2)
    ax.grid(True)
    # ax.set_xlabel("X offset (cm)")
    ax.set_ylabel("F1")
    ax2.set_ylabel("F2")
    # plt.tight_layout()
    plt.savefig(os.path.join(args.res_dir,"cba_X_offset.pdf"))
    # Y
    fig = plt.figure(figsize=args.fig_size)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data = res_sim3[Nrot*3+Ntsl:Nrot*3+Ntsl*2,:]
    f1 = data[:,0]
    f2 = data[:,1]
    C = data[:,2]
    nrev = C > args.he_threshold
    x = tsl_offset
    plot_f1 = ax.plot(x,f1,'b-',marker='.')
    plot_f2 = ax2.plot(x,f2,'r-',marker='.')
    plot_C = ax.plot(x[nrev],f1[nrev],'k^',markersize=7)
    ax2.plot(x[nrev],f2[nrev],'k^',markersize=7)
    plt.legend(plot_f1+plot_f2+plot_C,['F1','F2','IFeas'],loc='upper center',frameon=True,ncol=2)
    ax.grid(True)
    # ax.set_xlabel("Y offset (cm)")
    ax.set_ylabel("F1")
    ax2.set_ylabel("F2")
    # plt.tight_layout()
    plt.savefig(os.path.join(args.res_dir,"cba_Y_offset.pdf"))
    # Z
    fig = plt.figure(figsize=args.fig_size)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data = res_sim3[Nrot*3+Ntsl*2:Nrot*3+Ntsl*3,:]
    f1 = data[:,0]
    f2 = data[:,1]
    C = data[:,2]
    nrev = C > args.he_threshold
    x = tsl_offset
    plot_f1 = ax.plot(x,f1,'b-',marker='.')
    plot_f2 = ax2.plot(x,f2,'r-',marker='.')
    plot_C = ax.plot(x[nrev],f1[nrev],'k^',markersize=7)
    ax2.plot(x[nrev],f2[nrev],'k^',markersize=7)
    plt.legend(plot_f1+plot_f2+plot_C,['F1','F2','IFeas'],loc='upper center',frameon=True,ncol=2)
    ax.grid(True)
    # ax.set_xlabel("Z offset (cm)")
    ax.set_ylabel("F1")
    ax2.set_ylabel("F2")
    # plt.tight_layout()
    plt.savefig(os.path.join(args.res_dir,"cba_Z_offset.pdf"))
    # scale
    fig = plt.figure(figsize=args.fig_size)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    data = res_sim3[Nrot*3+Ntsl*3:Nrot*3+Ntsl*3+Nscale,:]
    f1 = data[:,0]
    f2 = data[:,1]
    C = data[:,2]
    nrev = C > args.he_threshold
    x = scale_offset
    plot_f1 = ax.plot(x,f1,'b-',marker='.')
    plot_f2 = ax2.plot(x,f2,'r-',marker='.')
    plot_C = ax.plot(x[nrev],f1[nrev],'k^',markersize=7)
    ax2.plot(x[nrev],f2[nrev],'k^',markersize=7)
    plt.legend(plot_f1+plot_f2+plot_C,['F1','F2','IFeas'],loc='upper center',frameon=True,ncol=2)
    ax.grid(True)
    # ax.set_xlabel("scale offset")
    ax.set_ylabel("F1")
    ax2.set_ylabel("F2")
    # plt.tight_layout()
    plt.savefig(os.path.join(args.res_dir,"cba_scale_offset.pdf"))