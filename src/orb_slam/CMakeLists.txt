cmake_minimum_required(VERSION 3.4)
project(ORB_SLAM2)
set(OpenGL_GL_PREFERENCE "GLVND")   # Set for Pangolin
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE(${PROJECT_NAME} " Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -fopenmp ")  # add -fopenmp for OpenMP
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
set(g2o_DIR "/usr/local/lib/g2o")

include_directories(${G2O_INCLUDE_DIRS} "/usr/include/eigen3")
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../../lib)

SET (G2O_LIBS cxsparse g2o_csparse_extension g2o_cli g2o_core g2o_interface g2o_parser g2o_solver_cholmod
g2o_solver_csparse g2o_solver_dense g2o_solver_pcg g2o_stuff g2o_types_icp
g2o_types_sba g2o_types_sim3 g2o_types_slam2d g2o_types_slam3d)
add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
)# src/PnPsolver.cc

include_directories(
  /usr/include/boost
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${G2O_LIBS}
boost_serialization
)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples)
# add_executable(kitti_orb ../examples/orb_kitti.cpp)
# target_link_libraries(kitti_orb
# ${PROJECT_NAME}
# /usr/local/lib/libglog.so  # ??
# )