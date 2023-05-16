#include "HECalib.h"
#include "NLHECalib.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include "kitti_tools.h"
#include <limits>
#include "orb_slam/include/System.h"
#include <yaml-cpp/yaml.h>
#include <vector>
#include <map>

int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "Got " << argc - 1 << " Parameters, but Expected parameters: yaml_file orb_setting_file orb_vocabulary_file" << std::endl;
        exit(0);
    }
    std::string config_file(argv[1]), ORBSetFilename(argv[2]), VocFilename(argv[3]);
    YAML::Node config = YAML::LoadFile(config_file);
    std::string base_dir = config["BaseDir"].as<std::string>();
    checkpath(base_dir);
    std::vector<Eigen::Isometry3d> vTwc, vTwl, vTwlraw;
    std::string TwcFilename = base_dir + config["VOFile"].as<std::string>();
    std::string TwlFilename = base_dir + config["LOFile"].as<std::string>();
    std::string resFileName = base_dir + config["ResFile"].as<std::string>();
    std::string KeyFrameDir = base_dir + config["KeyFrameDir"].as<std::string>();
    std::string KyeFrameIdFile = base_dir + config["VOIdFile"].as<std::string>();
    std::string MapFileNmae = base_dir + config["MapFile"].as<std::string>();
    std::map<std::string, int> method_options = config["method_options"].as<std::map<std::string, int> >();
    std::string method = config["method"].as<std::string>();
    assert(method_options.count(method)>0);
    std::cout << "Use " << method << " Bundle Adjustment for calibration." << std::endl;
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
    
    

    ORB_SLAM2::System VSLAM(VocFilename, ORBSetFilename, ORB_SLAM2::System::MONOCULAR, false);
    VSLAM.Shutdown(); // Do not need any thread
    VSLAM.RestoreSystemFromFile(KeyFrameDir, MapFileNmae);
    std::cout << "System Restored Successfully." << std::endl;
    std::tie(RCL,tCL,s) = HECalib(vmTwc, vmTwl);
    std::cout << "Ordinary Hand-eye Calibration:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "s :" << s << std::endl;

    std::tie(RCL,tCL,s) = HECalibRobustKernelg2o(vmTwc, vmTwl, RCL, tCL, s, true, 0.003);
    std::cout << "Robust Kernel Hand-eye Calibration with Regulation:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "s :" << s << std::endl;
    Eigen::Isometry3d rigidTCL = Eigen::Isometry3d::Identity();
    rigidTCL.rotate(RCL);
    rigidTCL.pretranslate(tCL);

    Eigen::Isometry3d TCL0(rigidTCL);
    int ngood;
    if(method == "global"){
        ngood = VSLAM.OptimizeExtrinsicGlobal(vTwl, TCL0, s);
        std::cout << "Bundle Adjustment Optimization (Global):\n";
        std::cout << "Rotation: \n" << TCL0.rotation() << std::endl;
        std::cout << "Translation: \n" << TCL0.translation() << std::endl;
        std::cout<< "scale=  "<< s <<"  "<<"ngood=  "<<ngood<<std::endl;
    }
    else
    {
        TCL0 = rigidTCL;
        ngood = VSLAM.OptimizeExtrinsicLocal(vTwl, TCL0, s);
        std::cout << "Bundle Adjustment Optimization (Local):\n";
        std::cout << "Rotation: \n" << TCL0.rotation() << std::endl;
        std::cout << "Translation: \n" << TCL0.translation() << std::endl;
        std::cout<< "scale=  "<< s <<"  "<<"ngood=  "<< ngood <<std::endl;
    }
    writeSim3(resFileName, TCL0, s);
    std::cout << "Result of BA Calibration saved to " << resFileName << std::endl;



    // std::tie(RCL,tCL,s) = HECalibRobustKernelg2o(vmTwc, vmTwl, RCL, tCL, s);
    // std::cout << "Robust Hand-eye Calibration:\n";
    // std::cout << "Rotation: \n" << RCL << std::endl;
    // std::cout << "Translation: \n" << tCL << std::endl;
    // std::cout << "s :" << s << std::endl;

    // std::tie(RCL,tCL,s) = HECalibLPg2o(vmTwc, vmTwl, RCL, tCL, s);
    // std::cout << "Line Process Hand-eye Calibration:\n";
    // std::cout << "Rotation: \n" << RCL << std::endl;
    // std::cout << "Translation: \n" << tCL << std::endl;
    // std::cout << "s :" << s << std::endl;
}