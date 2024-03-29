cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp")
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "Compiling with OpenMP ${OpenMP_C_FLAGS}" )
endif(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fopenmp")  # -g -fsanitize=address
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
MESSAGE(${PROJECT_NAME} " Build type: " ${CMAKE_BUILD_TYPE})

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(src/floam)
add_subdirectory(src/orb_slam)
add_subdirectory(src/scancontext)

# g2o
set(g2o_DIR "/usr/local/lib")
set(NOMAD_LIB_DIRS
"/data/Pacakge/nomad/build/release/lib"
"/data/Pacakge/nomad/src")
set(NOMAD_INCLUDE_DIR "/data/Pacakge/nomad/build/release/include/nomad")
set(IGL_INCLUDE_DIR
"/data/Pacakge/libigl/include"
)
set(IGL_VIEWER_INCLUDE_DIR "/data/Pacakge/libigl/build/_deps/glfw-src/deps")
set(IGL_LIB_DIR "/data/Pacakge/libigl/build/lib")
set(GLOG_LIB /usr/local/lib/libglog.so)
set(YAML_CPP_LIB /usr/local/lib/libyaml-cpp.so)
link_libraries(${YAML_CPP_LIB})
include_directories(${G2O_INCLUDE_DIRS} "/usr/include/eigen3")
link_directories(${g2o_DIR} ${NOMAD_LIB_DIRS} ${IGL_LIB_DIR})

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
find_package(GTSAM REQUIRED QUIET)
find_package(Open3D HINTS REQUIRD)
list(APPEND Open3D_LIBRARIES dl)
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")
    link_directories(${Open3D_LIBRARY_DIRS})
endif()
include_directories(
  include
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)
set(G2O_LIBS
g2o_csparse_extension
g2o_core 
g2o_solver_cholmod
g2o_solver_csparse
g2o_solver_dense
g2o_solver_pcg 
g2o_stuff
g2o_types_icp
g2o_types_sba
g2o_types_sim3
g2o_types_slam2d
g2o_types_slam3d)
set(NOMAD_LIBS
nomadAlgos
nomadUtils 
nomadEval
)
set(IO_LIBS
${OpenCV_LIBS}
${PCL_LIBRARIES}
)
set(GMP_LIB /usr/lib/x86_64-linux-gnu/libgmp.so)

set(IGL_LIBS
CoMISo
glad
glfw3
igl
igl_copyleft_cgal
igl_copyleft_comiso
igl_copyleft_core
igl_copyleft_tetgen
igl_embree
igl_glfw
igl_imgui
igl_opengl
igl_png
igl_predicates
igl_restricted_triangle
igl_xml
imgui
imguizmo
predicates
stb
tetgen
tinyxml2
triangle
${GMP_LIB}
)


add_executable(ba_demo src/ba_demo.cpp)
target_link_libraries(ba_demo ${OpenCV_LIBS} ${G2O_LIBS})

add_executable(ba_demo_data src/ba_demo_data.cpp)
target_link_libraries(ba_demo_data
${OpenCV_LIBS}
${G2O_LIBS}
${GLOG_LIB}  # ??
)

add_library(backend SHARED src/backend_opt.cpp)
add_dependencies(backend scancontext)
target_link_libraries(backend
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  scancontext
  gtsam
  ${Open3D_LIBRARIES}
)
target_include_directories(backend PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)



set(ORB_INCLUDE_DIRS
${PROJECT_SOURCE_DIR}/src/orb_slam
)

set(ORB_LIBARIES
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${G2O_LIBS}
${PROJECT_SOURCE_DIR}/lib/libORB_SLAM2.so
${GLOG_LIB}  # ??
)

set(FLOAM_LIBRARIES
${PROJECT_SOURCE_DIR}/lib/libfloam.so
${PCL_LIBRARIES}
)

add_executable(floam_run src/examples/floam_kitti.cpp)
add_dependencies(floam_run floam)
target_link_libraries(floam_run
${FLOAM_LIBRARIES}
)

add_executable(floam_backend
src/examples/floam_backend.cpp
)
add_dependencies(floam_backend backend)
target_link_libraries(floam_backend
  backend
)

add_executable(orb_run src/examples/orb_kitti.cpp)
add_dependencies(orb_run ORB_SLAM2)
target_link_libraries(orb_run
${ORB_LIBARIES}
)
target_include_directories(orb_run PRIVATE 
${ORB_INCLUDE_DIRS}
)

add_executable(orb_store src/examples/orb_kitti_store.cpp)
add_dependencies(orb_store ORB_SLAM2)
target_link_libraries(orb_store
${ORB_LIBARIES}
${PCL_LIBRARIES}
)
target_include_directories(orb_store PRIVATE 
${ORB_INCLUDE_DIRS}
)

add_executable(orb_restore src/examples/orb_restore.cpp)
add_dependencies(orb_restore ORB_SLAM2)
target_link_libraries(orb_restore
${ORB_LIBARIES}
)
target_include_directories(orb_restore PRIVATE 
${ORB_INCLUDE_DIRS}
)

add_executable(orb_save_map src/examples/orb_save_map.cpp)
add_dependencies(orb_save_map ORB_SLAM2)
target_link_libraries(orb_save_map
${ORB_LIBARIES}
${Open3D_LIBRARIES}
)
target_include_directories(orb_save_map PRIVATE 
${ORB_INCLUDE_DIRS}
)
# add_executable(kitti_union src/examples/union_kitti.cpp)
# add_dependencies(kitti_union floam ORB_SLAM2)
# target_link_libraries(kitti_union
# ${FLOAM_LIBRARIES}
# ${ORB_LIBARIES}
# )
# target_include_directories(kitti_union PRIVATE 
# include
# ${ORB_INCLUDE_DIRS}
# )

add_executable(ba_calib src/examples/ba_calib.cpp)
add_dependencies(ba_calib floam ORB_SLAM2)
target_link_libraries(ba_calib
${ORB_LIBARIES}
)



add_executable(he_calib src/examples/he_calib.cpp)
target_link_libraries(
  he_calib
  ${EIGEN3_LIBS}
  ${G2O_LIBS}
  ${GLOG_LIB} 
)

add_executable(icp_calib src/examples/icp_calib.cpp)
target_link_libraries(icp_calib
${Open3D_LIBRARIES}
)



# add_executable(iba_calib_test src/examples/iba_calib_test.cpp)
# add_dependencies(iba_calib_test ORB_SLAM2)
# target_link_libraries(iba_calib_test
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# )
# target_include_directories(iba_calib_test PRIVATE 
# ${ORB_INCLUDE_DIRS}
# )

# add_executable(iba_calib_plane src/examples/iba_calib_plane.cpp)
# add_dependencies(iba_calib_plane ORB_SLAM2)
# target_link_libraries(iba_calib_plane
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# )
# target_include_directories(iba_calib_plane PRIVATE 
# ${ORB_INCLUDE_DIRS}
# )

# add_executable(iba_calib_plane2 src/examples/iba_calib_plane2.cpp)
# add_dependencies(iba_calib_plane2 ORB_SLAM2)
# target_link_libraries(iba_calib_plane2
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# ${CERES_LIBRARIES}
# )
# target_include_directories(iba_calib_plane2 PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
# )

# add_executable(iba_calib_gpr2 src/examples/iba_calib_gpr2.cpp)
# add_dependencies(iba_calib_gpr2 ORB_SLAM2)
# target_link_libraries(iba_calib_gpr2
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# ${CERES_LIBRARIES}
# )
# target_include_directories(iba_calib_gpr2 PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
# )

# add_executable(iba_calib3d_gpr src/examples/iba_calib3d_gpr.cpp)
# add_dependencies(iba_calib3d_gpr ORB_SLAM2)
# target_link_libraries(iba_calib3d_gpr
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# ${CERES_LIBRARIES}
# )
# target_include_directories(iba_calib3d_gpr PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
# )

# add_executable(iba_calib3dq_gpr src/examples/iba_calib3dq_gpr.cpp)
# add_dependencies(iba_calib3dq_gpr ORB_SLAM2)
# target_link_libraries(iba_calib3dq_gpr
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# ${CERES_LIBRARIES}
# ${Open3D_LIBRARIES}
# ${IGL_LIBS}
# )
# target_include_directories(iba_calib3dq_gpr PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
# ${IGL_INCLUDE_DIR}
# )

# add_executable(uiba_local src/examples/uiba_local.cpp)
# add_dependencies(uiba_local ORB_SLAM2)
# target_link_libraries(uiba_local
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# ${CERES_LIBRARIES}
# )
# target_include_directories(uiba_local PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
# )


add_executable(iba_local src/examples/iba_local.cpp)
add_dependencies(iba_local ORB_SLAM2)
target_link_libraries(iba_local
${ORB_LIBARIES}
${PCL_LIBRARIES}
${G2O_LIBS}
${GLOG_LIB} 
${CERES_LIBRARIES}
)
target_include_directories(iba_local PRIVATE 
${ORB_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS}
)

add_executable(iba_global src/examples/iba_global.cpp)
add_dependencies(iba_global ORB_SLAM2)
target_link_libraries(iba_global
${ORB_LIBARIES}
${PCL_LIBRARIES}
${G2O_LIBS}
${GLOG_LIB}
${NOMAD_LIBS}
)
target_include_directories(iba_global PRIVATE 
${ORB_INCLUDE_DIRS}
${NOMAD_INCLUDE_DIR}
)

add_executable(iba_func src/examples/iba_func.cpp)
add_dependencies(iba_func ORB_SLAM2)
target_link_libraries(iba_func
${ORB_LIBARIES}
${PCL_LIBRARIES}
${G2O_LIBS}
${GLOG_LIB}
)
target_include_directories(iba_func PRIVATE 
${ORB_INCLUDE_DIRS}
)

# add_executable(iba_global_bias src/examples/iba_global_bias.cpp)
# add_dependencies(iba_global_bias ORB_SLAM2)
# target_link_libraries(iba_global_bias
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB}
# ${NOMAD_LIBS}
# )
# target_include_directories(iba_global_bias PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${NOMAD_INCLUDE_DIR}
# )

# add_executable(iba_global_split src/examples/iba_global_split.cpp)
# add_dependencies(iba_global_split ORB_SLAM2)
# target_link_libraries(iba_global_split
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB}
# ${NOMAD_LIBS}
# )
# target_include_directories(iba_global_split PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${NOMAD_INCLUDE_DIR}
# )

# add_executable(uiba_calib_gpr src/examples/uiba_calib_gpr.cpp)
# add_dependencies(uiba_calib_gpr ORB_SLAM2)
# target_link_libraries(uiba_calib_gpr
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# ${CERES_LIBRARIES}
# )
# target_include_directories(uiba_calib_gpr PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
# ${IGL_INCLUDE_DIR}
# ${IGL_VIEWER_INCLUDE_DIR}
# )

# add_executable(uiba_calib_plane src/examples/uiba_calib_plane.cpp)
# add_dependencies(uiba_calib_plane ORB_SLAM2)
# target_link_libraries(uiba_calib_plane
# ${ORB_LIBARIES}
# ${PCL_LIBRARIES}
# ${G2O_LIBS}
# ${GLOG_LIB} 
# ${CERES_LIBRARIES}
# )
# target_include_directories(uiba_calib_plane PRIVATE 
# ${ORB_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
# )


# add_executable(test_gpr src/test/test_gpr.cpp)
# target_link_libraries(test_gpr
# ${GLOG_LIB}
# ${CERES_LIBRARIES}
# )


# add_executable(test_igl src/test/test_igl.cpp)
# target_link_libraries(test_igl
# ${IO_LIBS}
# ${Open3D_LIBRARIES}
# ${IGL_LIBS}
# )

# target_include_directories(test_igl PRIVATE 
# ${IGL_INCLUDE_DIR}
# ${IGL_VIEWER_INCLUDE_DIR}
# )