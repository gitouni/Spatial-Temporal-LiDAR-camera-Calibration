#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <yaml-cpp/yaml.h>
#include <vector>
#include "HECalib.h"
#include "NLHECalib.hpp"
#include "kitti_tools.h"

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
    const std::string TwcFilename = base_dir + io_config["VOFile"].as<std::string>();
    const std::string TwlFilename = base_dir + io_config["LOFile"].as<std::string>();
    const std::string resFileName = base_dir + io_config["ResFile"].as<std::string>();
    const std::string resLPFileName = base_dir + io_config["ResLPFile"].as<std::string>();
    const std::string KyeFrameIdFile = base_dir + io_config["VOIdFile"].as<std::string>();

    const bool regulation = runtime_config["regulation"].as<bool>();
    const double regulation_weight = runtime_config["regulation_weight"].as<double>();
    const int inner_iter = runtime_config["inner_iter"].as<int>();
    const int ex_iter = runtime_config["ex_iter"].as<int>();
    const double init_mu = runtime_config["init_mu"].as<double>();
    const double min_mu = runtime_config["min_mu"].as<double>();
    const double divide_factor = runtime_config["divide_factor"].as<double>();
    const bool verborse = runtime_config["verborse"].as<bool>();

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
    writeSim3(resFileName, RCL, tCL, s);
    
    std::cout << "Result of Hand-eye Calibration saved to " << resFileName << std::endl;
    std::tie(RCL,tCL,s) = HECalibLineProcessg2o(vmTwc, vmTwl, RCL, tCL, s, inner_iter, init_mu, divide_factor, min_mu, ex_iter, regulation, regulation_weight, verborse);
    std::cout << "Line Process Hand-eye Calibration:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "s :" << s << std::endl;
    writeSim3(resLPFileName, RCL, tCL, s);
}