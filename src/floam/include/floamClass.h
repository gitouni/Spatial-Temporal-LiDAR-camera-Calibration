#pragma once

#include "lidar.h"
#include "laserMappingClass.h"
#include "laserProcessingClass.h"
#include "lidarOptimization.h"
#include "odomEstimationClass.h"


namespace FLOAM{
    typedef pcl::PointXYZI PointType;
    enum eLiDARType{
        VLP_16=0,
        HDL_32=1,
        HDL_64=2
    };


    class System{
        public:
            System(int num_lines=64, double scan_period=0.1,double max_dis=90,double min_dis=3.0,double map_resolution=0.4);
            System(eLiDARType type, double scan_period=0.1, double max_dis=90);
            Eigen::Isometry3d Track(const pcl::PointCloud<PointType>::Ptr pointcloud_in);
            lidar::Lidar *lidar_params;
            bool odom_init;
        private:
            LaserProcessingClass *laserprocess;
            OdomEstimationClass *odomEstimation;
            LaserMappingClass *laserMapping;
    };

}