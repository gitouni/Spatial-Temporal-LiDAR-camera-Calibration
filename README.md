# Spatial-Temporal-LiDAR-camera-Calibration
Spatial-Temporal-LiDAR-camera-Calibration
# Performance on KITTI Odometry
* ```rotation units```: degree
* ```translation units```: cm


|Sequence| X| Y | Z | roll | pitch | yaw |
|---|---|---|---|---|---|---|
|00|0.49|1.33|-0.52|-0.049|-0.225|-0.155|
|02|-2.80|-1.63|1.18|-0.01|-0.246|-0.159|
|03|-3.25|3.23|1.21|-0.083|-0.096|-0.087|
|04|1.22|0.89|2.83|0.011|-0.088|0.074|
|05|-0.94|-0.52|1.27|0.190|-0.128|-0.181|

# Tested Environment
|C++|CMake|g++|python|System|
|---|---|---|---|---|
|C++ 17| CMake 3.25| 9.4.0| 3.8| Ubuntu 20.04|

More recent version would be OK. 
# Dependencies
* [Eigen 3](http://eigen.tuxfamily.org/) (`libeigen3-dev`)
* [Ceres-Solver](http://ceres-solver.org/)
* [OpenMP](https://github.com/llvm-mirror/openmp) (`libomp-dev`)
* [OpenCV 4.x](http://opencv.org/)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin) (visual tools for ORB_SLAM2)
* [g2o 2023](https://github.com/RainerKuemmerle/g2o/releases/tag/20230223_git) (`-DG2O_USE_OPENMP=ON`, NO support for earlier version)
* [Open3D](https://github.com/isl-org/Open3D) (built for C++)
# Install
* Copy [Thirdparty](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Thirdparty) and [Vocabulary](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Vocabulary) directories to [src/orb_slam/](./src/orb_slam/) <details>
  <summary>Troubleshooting</summary>
  Note that the implementation of ORB_SLAM2 in our repo is different from the original one, so DO NOT copy the whole ORB_SLAM2 repo to replace our directory</details>
* Build and Compile: `cd build && cmake .. -Wno-dev && make -j` <details>
  <summary>TroubleShooting</summary>
  If you have installed g2o through ROS (if you have ROS packages like `base_local_planner`/`teb_local_planner`/`mpc_local_planner`), please exclude it from    LD_LIBRARY_PATH environment variable, or `source config/settings.sh`. </details>
* For performance, build/rebuild g2o with CMake flag `-DG2O_USE_OPENMP=ON`. See [here](https://github.com/RainerKuemmerle/g2o/issues/689#issuecomment-1565658030).
## Step 1: Estimate Camera and Lidar Poses
### orb_slam
an example command for KITTI 04 Sequence:
```
./orb_store ../data/04/ ../config/orb_ori/KITTI04-12.yaml ../data/Vocabulary/ORBvoc.txt ../KITTI-04/slam_res/Twc.txt ../KITTI-04/KeyFrames/ ../KITTI-04/Map.yml 1.5
```
* `../data/04/`: sequence directory of KITTI Sequence 04
* `../config/orb_ori/KITTI04-12.yaml`: yaml file for ORB_SLAM
* `../data/Vocabulary/ORBvoc.txt`: DBoW2 Vocabulary txt file for ORB_SLAM
* `../KITTI-04/slam_res/Twc.txt`: KeyFrame poses saved by ORB_SLAM, in KITTI format
* ```../KITTI-04/KeyFrames/```: KeyFrame information saved by ORB_SLAM, can be restored during runtime
* ```../KITTI-04/Map.yml```: Map saved by ORB_SLAM, can be restored during runtime
* ```1.5```: Slow Rate of ORB_SLAM2. Duration between Frames: computation + wait_time >= 1.5 * real timestamp
### F-LOAM
an example command for KITTI 04 Sequence:
```
./floam_run ../data/04/velodyne/ ../KITTI-04/slam_res/floam_raw_04.txt
```
* ```../data/04/velodyne/```: directory containing the Lidar PointClouds, whose filenames must be sorted by timestamp.
* ```../KITTI-04/slam_res/floam_raw_04.txt```: Lidar Poses Estimated by F-LOAM. 
<details><summary>Note</summary>Note that the number of Lidar Poses is not equal to Camera poses because ORB_SLAM only saved KeyFrame Poses. However, the File Id (FrameId) of these KeyFrames are saved to 'FrameId.yml' in the same directory of 'Map.yml'</details>

### F-LOAM Backend Optimization
an example command for KITTI 04 Sequence:
```
 ./floam_backend ../config/loam/backend.yml ../KITTI-04/slam_res/floam_raw_08.txt ../data/04/velodyne/
 ```
 * ```../config/loam/backend.yml```: config file of backend optimzation
 * ```../KITTI-04/slam_res/floam_raw_08.txt```: Lidar poses estimated by F-LOAM
 * ```../data/04/velodyne/```: directory containing the Lidar PointClouds, whose filenames must be sorted by timestamp.
