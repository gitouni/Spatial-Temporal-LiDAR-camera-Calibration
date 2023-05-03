#include "floamClass.h"

namespace FLOAM{
    System::System(int num_lines, double scan_period, double max_dis, double min_dis, double map_resolution):
    odom_init(false), lidar_params(new lidar::Lidar),
    laserprocess(new LaserProcessingClass), odomEstimation(new OdomEstimationClass),laserMapping(new LaserMappingClass){
        lidar_params->setLines(num_lines);
        lidar_params->setScanPeriod(scan_period);
        lidar_params->setMaxDistance(max_dis);
        lidar_params->setMinDistance(min_dis);
        laserprocess->init(*lidar_params);
        odomEstimation->init(*lidar_params, map_resolution);
    }
    System::System(eLiDARType type, double scan_period, double max_dis):
    odom_init(false), lidar_params(new lidar::Lidar),
    laserprocess(new LaserProcessingClass), odomEstimation(new OdomEstimationClass),laserMapping(new LaserMappingClass){
        short num_lines;
        double min_dis;
        double map_resolution;
        if (type==VLP_16){
            num_lines = 16;
            min_dis = 0.3;
            map_resolution = 0.2;
        }
        else if (type==HDL_32)
        {
            num_lines = 32;
            min_dis = 0.5;
            map_resolution = 0.2;
        }
        else if (type==HDL_64)
        {
            num_lines = 64;
            min_dis = 3.0;
            map_resolution = 0.4;
        }else{
            throw std::invalid_argument("Invalid LiDARType");
            return;
        }
        lidar_params->setLines(num_lines);
        lidar_params->setScanPeriod(scan_period);
        lidar_params->setMaxDistance(max_dis);
        lidar_params->setMinDistance(min_dis);
        laserprocess->init(*lidar_params);
        odomEstimation->init(*lidar_params, map_resolution);
    }
    Eigen::Isometry3d System::Track(const pcl::PointCloud<PointType>::Ptr pointcloud_in){
        pcl::PointCloud<PointType>::Ptr pointcloud_edge(new pcl::PointCloud<PointType>);          
        pcl::PointCloud<PointType>::Ptr pointcloud_surf(new pcl::PointCloud<PointType>);
        laserprocess->featureExtraction(pointcloud_in, pointcloud_edge, pointcloud_surf);
        if(!odom_init){
            odomEstimation->initMapWithPoints(pointcloud_edge, pointcloud_surf);
            odom_init = true;
            return Eigen::Isometry3d::Identity();
        }
        else{
            odomEstimation->updatePointsToMap(pointcloud_edge, pointcloud_surf);
            return odomEstimation->odom;
        }
    }

}