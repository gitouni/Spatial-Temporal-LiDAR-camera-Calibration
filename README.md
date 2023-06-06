# Spatial-Temporal-LiDAR-camera-Calibration
Spatial-Temporal-LiDAR-camera-Calibration
# Tested Environment
|C++|CMake|g++|python|System|
|---|---|---|---|---|
|C++ 17| CMake 3.20| 9.4.0| 3.8| Ubuntu 20.04|

More recent version would be OK. 
# Dependencies
* [Eigen 3](http://eigen.tuxfamily.org/) (`libeigen3-dev`)
* [LBFGSpp](https://github.com/yixuan/LBFGSpp) (GPR hyperparameter adaptation)
* [OpenMP](https://github.com/llvm-mirror/openmp) (`libomp-dev`)
* [OpenCV 4.x](http://opencv.org/)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin) (visual tools for ORB_SLAM2)
* [g2o 2023](https://github.com/RainerKuemmerle/g2o/releases/tag/20230223_git) (compatible with C++ 17)
* [Open3D](https://github.com/isl-org/Open3D) (built for C++)
# Install
* Copy [Thirdparty](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Thirdparty) and [Vocabulary](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Vocabulary) directories to [src/orb_slam/](./src/orb_slam/) <details>
  <summary>Troubleshooting</summary>
  Note that the implementation of ORB_SLAM2 in our repo is different from the original one, so DO NOT copy the whole ORB_SLAM2 repo to replace our directory</details>
* Build and Compile: `cd build && cmake .. -Wno-dev && make -j` <details>
  <summary>TroubleShooting</summary>
  If you have installed g2o through ROS (if you have ROS packages like `base_local_planner`/`teb_local_planner`/`mpc_local_planner`), please exclude it from    LD_LIBRARY_PATH environment variable, or `source config/settings.sh`. </details>
* For performance, rebuild g2o with CMake flag `-DG2O_USE_OPENMP=ON`. See [here](https://github.com/RainerKuemmerle/g2o/issues/689#issuecomment-1565658030).
# How to Use
## Step 1
