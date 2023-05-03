#include "HECalib.h"
#include "NLHECalib.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include "kitti_tools.h"
#include <limits>


int main(int argc, char** argv){
    if(argc < 4){
        std::cout << "Got " << argc - 1 << " Parameters, but expected 3";
        throw std::invalid_argument("Expected parameters: Twc_file Twl_file res_file");
        exit(0);
    }
    std::vector<Eigen::Isometry3d> vTwc, vTwl;
    std::string TwcFilename(argv[1]), TwlFilename(argv[2]), resFileName(argv[3]);
    ReadPoseList(TwcFilename, vTwc);
    ReadPoseList(TwlFilename, vTwl);
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

    Eigen::Isometry3d rigidTCL = Eigen::Isometry3d::Identity();
    rigidTCL.rotate(RCL);
    rigidTCL.pretranslate(tCL);
    
    writePose(resFileName, rigidTCL, s);
    std::cout << "Result of Hand-eye Calibration saved to " << resFileName << std::endl;
    std::tie(RCL,tCL,s) = HECalibRobustKernelg2o(vmTwc, vmTwl, RCL, tCL, s);
    std::cout << "Robust Hand-eye Calibration:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "s :" << s << std::endl;

    std::tie(RCL,tCL,s) = HECalibLPg2o(vmTwc, vmTwl, RCL, tCL, s);
    std::cout << "Line Process Hand-eye Calibration:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "s :" << s << std::endl;
}