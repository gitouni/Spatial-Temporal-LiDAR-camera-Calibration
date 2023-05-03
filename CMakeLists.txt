cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
MESSAGE(${PROJECT_NAME} " Build type: " ${CMAKE_BUILD_TYPE})

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(src/floam)
add_subdirectory(src/orb_slam)


# g2o
set(g2o_DIR "/usr/local/lib/g2o")
set(GLOG_LIBRARY_DIR /usr/local/lib/libglog.so)
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIRS} "/usr/include/eigen3")
SET (G2O_LIBS g2o_csparse_extension g2o_core g2o_solver_cholmod
g2o_solver_csparse g2o_solver_dense g2o_solver_pcg g2o_stuff g2o_types_icp
g2o_types_sba g2o_types_sim3 g2o_types_slam2d g2o_types_slam3d)
# OpenCV
find_package(OpenCV 4 REQUIRED)
# Eigen3
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)


include_directories(
  include
  ${PROJECT_SOURCE_DIR}/src
)


add_executable(ba_demo src/ba_demo.cpp)
target_link_libraries(ba_demo ${OpenCV_LIBS} ${G2O_LIBS})

add_executable(ba_demo_data src/ba_demo_data.cpp)
target_link_libraries(ba_demo_data
${OpenCV_LIBS}
${G2O_LIBS}
${GLOG_LIBRARY_DIR}  # ??
)

set(ORB_INCLUDE_DIRS
${PROJECT_SOURCE_DIR}/src/orb_slam
)

set(ORB_LIBARIES
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${G2O_LIBS}
${PROJECT_SOURCE_DIR}/src/orb_slam/lib/libORB_SLAM2.so
${GLOG_LIBRARY_DIR}  # ??
)

set(FLOAM_LIBRARIES
${PROJECT_SOURCE_DIR}/src/floam/lib/libfloam.so
${PCL_LIBRARIES}
)

add_executable(kitti_floam src/examples/floam_kitti.cpp)
add_dependencies(kitti_floam floam)
target_link_libraries(kitti_floam
${FLOAM_LIBRARIES}
)


add_executable(kitti_orb src/examples/orb_kitti.cpp)
add_dependencies(kitti_orb ORB_SLAM2)
target_link_libraries(kitti_orb
${ORB_LIBARIES}
)
target_include_directories(kitti_orb PRIVATE 
${ORB_INCLUDE_DIRS}
)

add_executable(kitti_orb_restore src/examples/orb_restore.cpp)
add_dependencies(kitti_orb_restore ORB_SLAM2)
target_link_libraries(kitti_orb_restore
${ORB_LIBARIES}
)
target_include_directories(kitti_orb_restore PRIVATE 
${ORB_INCLUDE_DIRS}
)

add_executable(kitti_union src/examples/union_kitti.cpp)
add_dependencies(kitti_union floam ORB_SLAM2)
target_link_libraries(kitti_union
${FLOAM_LIBRARIES}
${ORB_LIBARIES}
)
target_include_directories(kitti_union PRIVATE 
${ORB_INCLUDE_DIRS}
)

add_executable(ba_calib src/examples/ba_calib.cpp)
add_dependencies(ba_calib floam ORB_SLAM2)
target_link_libraries(ba_calib
${ORB_LIBARIES}
)
target_include_directories(ba_calib PRIVATE 
${ORB_INCLUDE_DIRS}
)

add_executable(kitti_he_demo src/examples/he_demo_kitti.cpp)
target_link_libraries(
  kitti_he_demo
  ${EIGEN3_LIBS}
  ${G2O_LIBS}
  ${GLOG_LIBRARY_DIR} 
)
