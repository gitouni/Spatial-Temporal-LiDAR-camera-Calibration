import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
from matplotlib import pyplot as plt
from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D
import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ax_label_fontsize",type=int,default=16)
    parser.add_argument("--ax_legend_fontsize",type=int,default=16)
    parser.add_argument("--traj_name",type=str,default="../KITTI-07/slam_res/floam_isam_07.txt")
    parser.add_argument("--plot_mode",type=str,choices=["xy","xz"],default="xy")
    parser.add_argument("--plot_mark",type=str,default="r-")
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    parameters = {"xtick.labelsize":args.ax_label_fontsize,
                    "ytick.labelsize":args.ax_label_fontsize,
                    "legend.fontsize":args.ax_legend_fontsize}
    plt.rcParams.update(parameters)
    file2label = dict(groundtruth="GT",mono_with_sc="proposed",stereo_no_loop="stereo")
    marks = dict(groundtruth="k-",mono_with_sc="b-.",stereo_no_loop="r--")
    traj:PoseTrajectory3D = file_interface.read_kitti_poses_file(args.traj_name)
    if args.plot_mode == "xy":
        x = traj.positions_xyz[:,0]  # x
        y = traj.positions_xyz[:,1]  # y
    else:
        x = traj.positions_xyz[:,0]  # x
        y = traj.positions_xyz[:,2]  # z
    plt.figure()
    plt.plot(x,y,args.plot_mark)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()