# Spatial-Temporal-LiDAR-camera-Calibration
Spatial-Temporal-LiDAR-camera-Calibration
# Tested Environment
|C++|CMake|g++|python|System|
|---|---|---|---|---|
|C++ 17| CMake 3.20| 9.4.0| 3.8| Ubuntu 20.04|

More recent version would be OK. 
# Dependencies
* [Eigen 3](http://eigen.tuxfamily.org/) (`libeigen3-dev` by apt on Ubuntu/Debian Systems)
* [OpenCV 4.x](http://opencv.org/) (It should be installed with ROS noetic, but our codes do not need ROS)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin) (It is installed to view ORB_SLAM2, but does not impact our implementation)
* [g2o 2023](https://github.com/RainerKuemmerle/g2o/releases/tag/20230223_git) (It is compatible with C++ 17)
* [Open3D](https://github.com/isl-org/Open3D) (C++ build)
# Install
* Copy [Thirdparty](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Thirdparty) and [Vocabulary](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Vocabulary) directories to [src/orb_slam/](./src/orb_slam/) <details>
  <summary>Troubleshooting</summary>
  Note that this version of ORB_SLAM2 is compiled in C++ 17 version and g2o 2023, so DO NOT copy the whole ORB_SLAM2 repo to replace our directory</details>
* If you have installed g2o through ROS (if you have ROS packages like `base_local_planner`/`teb_local_planner`/`mpc_local_planner`), please exclude it from LD_LIBRARY_PATH environment variable. Or, you can simply source our shell in the terminal:`source config/settings.sh`
* Build and Compile: `cd build && cmake .. -Wno-dev && make -j`

# How to Use
## Step 1
