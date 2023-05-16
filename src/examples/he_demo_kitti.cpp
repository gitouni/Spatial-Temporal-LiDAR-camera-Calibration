#include "HECalib.h"
#include "NLHECalib.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include "kitti_tools.h"
#include <limits>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <map>

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Got " << argc - 1 << " Parameters, but Expected parameters: yaml_file" << std::endl;
        exit(0);
    }
    std::string config_file(argv[1]);
    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node io_config = config["io"];
    YAML::Node runtime_config = config["runtime"];
    std::string base_dir = io_config["BaseDir"].as<std::string>();
    checkpath(base_dir);
    std::vector<Eigen::Isometry3d> vTwc, vTwl, vTwlraw;
    std::string TwcFilename = base_dir + io_config["VOFile"].as<std::string>();
    std::string TwlFilename = base_dir + io_config["LOFile"].as<std::string>();
    std::string resFileName = base_dir + io_config["ResFile"].as<std::string>();
    std::string KyeFrameIdFile = base_dir + io_config["VOIdFile"].as<std::string>();

    bool regulation = runtime_config["regulation"].as<bool>();
    double regulation_weight = runtime_config["regulation_weight"].as<double>();
    bool verborse = runtime_config["verborse"].as<bool>();

    ReadPoseList(TwcFilename, vTwc);
    ReadPoseList(TwlFilename, vTwlraw);
    YAML::Node FrameIdCfg = YAML::LoadFile(KyeFrameIdFile);
    std::vector<int> vKFFrameId = FrameIdCfg["mnFrameId"].as<std::vector<int>>();
    for(auto &KFId:vKFFrameId)
        vTwl.push_back(vTwlraw[KFId]);
    vTwlraw.clear();
    std::cout << "Load " << vTwc.size() << " Twc Pose files." << std::endl;
    std::cout << "Load " << vTwl.size() << " Twl Pose files." << std::endl;
    assert(vTwc.size()==vTwl.size());
    std::vector<Eigen::Isometry3d> vmTwc, vmTwl;
    vmTwc.reserve(vTwc.size()-1);
    vmTwl.reserve(vTwl.size()-1);
    pose2Motion(vTwc, vmTwc);
    pose2Motion(vTwl, vmTwl);
    Eigen::Matrix3d RCL;
    Eigen::Vector3d tCL;
    double s;
    
    std::tie(RCL,tCL,s) = HECalib(vmTwc, vmTwl);
    std::cout << "Ordinary Hand-eye Calibration:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "s :" << s << std::endl;

    std::tie(RCL,tCL,s) = HECalibRobustKernelg2o(vmTwc, vmTwl, RCL, tCL, s, regulation, regulation_weight, verborse);
    std::cout << "Robust Kernel Hand-eye Calibration with Regulation:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "scale :" << s << std::endl;
    Eigen::Isometry3d rigidTCL = Eigen::Isometry3d::Identity();
    rigidTCL.rotate(RCL);
    rigidTCL.pretranslate(tCL);
    writeSim3(resFileName, rigidTCL, s);
    std::cout << "Result of Hand-eye Calibration saved to " << resFileName << std::endl;
}