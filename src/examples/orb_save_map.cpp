#include "orb_slam/include/System.h"
#include "orb_slam/include/Converter.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/io/PointCloudIO.h"
#include <yaml-cpp/yaml.h>

int main(int argc, char **argv){
    if(argc < 2)
    {
        std::cout << "\033[31;1m Got " << argc-1 << " Parameters, expect config_file.\033[0m" << std::endl;
        exit(0);
    }
    YAML::Node config = YAML::LoadFile(argv[1]);
    YAML::Node orb_config = config["orb_setting"];
    YAML::Node io_config = config["io"];
    std::string vocabulary_path = orb_config["vocabulary"].as<std::string>();
    std::string setting_path = orb_config["setting"].as<std::string>();
    ORB_SLAM2::System SLAM(vocabulary_path, setting_path,ORB_SLAM2::System::MONOCULAR,false);
    std::string KeyFrameDir = io_config["KeyFrameDir"].as<std::string>();
    std::string MapFile = io_config["MapFile"].as<std::string>();
    std::string SavePath = io_config["SavePath"].as<std::string>();
    SLAM.Shutdown();
    SLAM.RestoreSystemFromFile(KeyFrameDir, MapFile);
    std::vector<ORB_SLAM2::MapPoint*> MapPoints = SLAM.GetAllMapPoints(true);
    std::vector<Eigen::Vector3d> MapPointsData;
    for(auto MapPoint:MapPoints){
        cv::Mat mat = MapPoint->GetWorldPos();
        Eigen::Vector3d pointvec = ORB_SLAM2::Converter::toVector3d(mat);
        MapPointsData.push_back(pointvec);
    }
    open3d::geometry::PointCloud Map(MapPointsData);
    open3d::io::WritePointCloudOption option(false, false, true);
    open3d::io::WritePointCloud(SavePath, Map, option);

}