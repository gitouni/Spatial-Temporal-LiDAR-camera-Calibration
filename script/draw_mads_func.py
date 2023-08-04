import argparse
import numpy as np
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ax_label_fontsize",type=int,default=14)
    parser.add_argument("--ax_legend_fontsize",type=int,default=14)
    parser.add_argument("--ax_marker_size",type=int,default=10)
    parser.add_argument("--he_threshold",type=float,default=0.03)
    parser.add_argument("--data",type=str,default="../KITTI-04/log/feas.txt")
    parser.add_argument("--res_dir",type=str,default="../fig/04/")
    parser.add_argument("--fig_size",type=float,nargs=2,default=[6.4,4.8])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    parameters = {"xtick.labelsize":args.ax_label_fontsize,
                "ytick.labelsize":args.ax_label_fontsize,
                "font.size":args.ax_label_fontsize,
                "legend.fontsize":args.ax_legend_fontsize,
                "lines.markersize":args.ax_marker_size}
    plt.rcParams.update(parameters)
    fig = plt.figure(figsize=args.fig_size)
    data = np.loadtxt(args.data,usecols=(0,8,9,10))
    ifeas_mask = np.logical_or(data[:,-1]>0, data[:,-2]>0)
    feas_mask = np.logical_not(ifeas_mask)
    func = data[:,1][feas_mask]
    idx = data[:,0][feas_mask]
    plt.plot(idx,func,'r-',marker='.')
    plt.ylabel("loss")
    plt.xlabel("index")
    plt.xlim(right=1000*((idx[-1] // 1000) + 1))
    plt.tight_layout()
    plt.savefig("../fig/04/MADS_func.svg")
    