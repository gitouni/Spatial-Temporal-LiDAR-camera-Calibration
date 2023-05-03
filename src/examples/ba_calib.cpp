#include "HECalib.h"
#include "NLHECalib.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include "kitti_tools.h"
#include <limits>
#include "orb_slam/include/System.h"

int main(int argc, char** argv){
    if(argc < 8){
        std::cout << "Got " << argc - 1 << " Parameters, but expected 7";
        throw std::invalid_argument("Expected parameters: Twc_file Twl_file res_file orb_setting_file orb_vocabulary_file keyFrame_dir map_file");
        exit(0);
    }
    std::vector<Eigen::Isometry3d> vTwc, vTwl;
    std::string TwcFilename(argv[1]), TwlFilename(argv[2]), resFileName(argv[3]);
    std::string ORBSetFilename(argv[4]), VocFilename(argv[5]), KeyFrameDir(argv[6]), MapFileNmae(argv[7]);
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
    
    
    // writePose(resFileName, rigidTCL, s);
    // std::cout << "Result of Hand-eye Calibration saved to " << resFileName << std::endl;
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
    std::cout << "Robust Kernel Hand-eye Calibration:\n";
    std::cout << "Rotation: \n" << RCL << std::endl;
    std::cout << "Translation: \n" << tCL << std::endl;
    std::cout << "s :" << s << std::endl;
    Eigen::Isometry3d rigidTCL = Eigen::Isometry3d::Identity();
    rigidTCL.rotate(RCL);
    // rigidTCL.pretranslate(tCL);

    Eigen::Isometry3d TCL0(rigidTCL);
    int ngood;
    // ngood = VSLAM.OptimizeExtrinsicGlobal(vTwl, TCL0, s);
    // std::cout << "Bundle Adjustment Optimization (Global):\n";
    // std::cout << "Rotation: \n" << TCL0.rotation() << std::endl;
    // std::cout << "Translation: \n" << TCL0.translation() << std::endl;
    // std::cout<< "scale=  "<< s <<"  "<<"ngood=  "<<ngood<<std::endl;

    TCL0 = rigidTCL;
    ngood = VSLAM.OptimizeExtrinsicLocal(vTwl, TCL0, s);
    std::cout << "Bundle Adjustment Optimization (Local):\n";
    std::cout << "Rotation: \n" << TCL0.rotation() << std::endl;
    std::cout << "Translation: \n" << TCL0.translation() << std::endl;
    std::cout<< "scale=  "<< s <<"  "<<"ngood=  "<< ngood <<std::endl;

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