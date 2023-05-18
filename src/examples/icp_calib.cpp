#include "open3d/geometry/PointCloud.h"
#include "open3d/io/PointCloudIO.h"
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/visualization/visualizer/Visualizer.h>
#include <yaml-cpp/yaml.h>
#include <kitti_tools.h>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>
int main(int argc, char** argv)
{
    if(argc < 2){
        std::cerr << "Need config_file arg" << std::endl;
        exit(0);
    }
    YAML::Node config = YAML::LoadFile(argv[1]);
    YAML::Node io_config = config["io"];
    YAML::Node runtime_config = config["runtime"];
    YAML::Node vis_config = config["vis"];
    const std::string lidar_pcd_filename = io_config["lidar_pcd"].as<std::string>();
    const std::string cam_pcd_filename = io_config["cam_pcd"].as<std::string>();
    const std::string init_sim3_filename = io_config["init_sim3"].as<std::string>();
    const std::string FrameIdFile = io_config["FrameId"].as<std::string>();
    const std::string savePath = io_config["savePath"].as<std::string>();

    const double icp_corr_dist = runtime_config["max_corr_dist"].as<double>();
    const int icp_max_iter = runtime_config["max_iter"].as<int>();

    const bool use_vis = vis_config["use_vis"].as<bool>();
    const std::string vis_name = vis_config["name"].as<std::string>();
    const int height = vis_config["height"].as<int>();
    const int width = vis_config["width"].as<int>();
    const std::vector<float> cam_color = vis_config["cam_pcd_color"].as<std::vector<float> >();
    const std::vector<float> lidar_color = vis_config["lidar_pcd_color"].as<std::vector<float> >();

    YAML::Node FrameIdConfig = YAML::LoadFile(FrameIdFile);
    std::vector<int> FrameId = FrameIdConfig["mnFrameId"].as<std::vector<int> >();
    std::shared_ptr<open3d::geometry::PointCloud> LidarPtPtr(new open3d::geometry::PointCloud());
    std::shared_ptr<open3d::geometry::PointCloud> CamPtPtr(new open3d::geometry::PointCloud());
    open3d::io::ReadPointCloudOption option("auto", true, true, true);
    open3d::io::ReadPointCloud(lidar_pcd_filename, *LidarPtPtr, option);
    open3d::io::ReadPointCloud(cam_pcd_filename, *CamPtPtr, option);
    if(FrameId[0] != 0)
    {
        std::cout << "Camera Reference Frame: " << FrameId[0] << "(!=0), need transform to the reference" << std::endl;
        const std::string lidar_posefile = io_config["lidar_pose"].as<std::string>();
        std::vector<Eigen::Isometry3d> lidar_poses;
        ReadPoseList(lidar_posefile, lidar_poses);
        Eigen::Isometry3d refpose = lidar_poses[FrameId[0]];
        LidarPtPtr->Transform(refpose.inverse().matrix());
    }
    Eigen::Matrix4d init_sim3;
    double init_scale;
    std::tie(init_sim3, init_scale) = readSim3(init_sim3_filename);
    init_sim3.topLeftCorner(3,3) = init_sim3.topLeftCorner(3,3).transpose().eval();  // R^T
    init_sim3.topRightCorner(3,1) = -init_sim3.topLeftCorner(3,3) * init_sim3.topRightCorner(3,1); // -R^T * t
    std::cout << "Raw RCL:\n" << init_sim3.topLeftCorner(3,3) << std::endl;
    std::cout << "Raw tCL:\n" << init_sim3.topRightCorner(3,1).transpose().eval() << std::endl;
    std::cout << "Raw scale:" << init_scale << std::endl;
    init_sim3.topLeftCorner(3,3) *= init_scale;
    open3d::pipelines::registration::ICPConvergenceCriteria criteria;
    criteria.max_iteration_ = icp_max_iter;
    auto reg = open3d::pipelines::registration::RegistrationICP(
        *CamPtPtr, *LidarPtPtr, icp_corr_dist, init_sim3,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(true),criteria
    );
    Eigen::Matrix3d diag = reg.transformation_.topLeftCorner(3,3) * reg.transformation_.topLeftCorner(3,3).transpose();
    double scale = sqrt(diag(0,0));
    Eigen::Matrix4d se3(reg.transformation_);
    se3.topLeftCorner(3,3) /= scale;
    se3 = se3.inverse().eval();
    std::cout << "Tuned RCL:\n" << se3.topLeftCorner(3,3) << std::endl;
    std::cout << "Tuned tCL:\n" << se3.topRightCorner(3,1).transpose() << std::endl;
    std::cout << "Tuned scale:\n" << scale << std::endl;
    CamPtPtr->Transform(reg.transformation_);
    CamPtPtr->PaintUniformColor((Eigen::Vector3d() << cam_color[0], cam_color[1], cam_color[2]).finished());
    LidarPtPtr->PaintUniformColor((Eigen::Vector3d() << lidar_color[0], lidar_color[1], lidar_color[2]).finished());
    if(use_vis){
        open3d::visualization::Visualizer visualizer;
        visualizer.CreateVisualizerWindow(vis_name, width, height);
        visualizer.AddGeometry(CamPtPtr);
        visualizer.AddGeometry(LidarPtPtr);
        visualizer.Run();
    }
    writeSim3(savePath, se3, scale);
}