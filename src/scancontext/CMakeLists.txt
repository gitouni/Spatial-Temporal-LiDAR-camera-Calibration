cmake_minimum_required(VERSION 2.8.3)
project(scancontext)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
MESSAGE(${PROJECT_NAME} " Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

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

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../../lib)

aux_source_directory(${PROJECT_SOURCE_DIR} DIR_LIB_SRCS)
add_library(scancontext SHARED ${DIR_LIB_SRCS})
target_link_libraries(scancontext ${EIGEN3_LIBRARIES} ${PCL_LIBRARIES})

target_include_directories(scancontext PRIVATE
  include
  ${PROJECT_SOURCE_DIR}/../
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)