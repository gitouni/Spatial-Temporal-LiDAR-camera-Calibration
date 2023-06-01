#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <thread>
#include <chrono>
#include <fstream>
#include <limits>
#include "io_tools.h"
#include "orb_slam/include/System.h"
#include <yaml-cpp/yaml.h>


void check_exist(std::string &file){
    if(!file_exist(file)){
        throw std::invalid_argument(file + " does not exist.");
        return;
    }
}

void writeData(std::string &outfile, std::vector<Eigen::Isometry3d> Tlist){
    std::ofstream of(outfile, std::ios::ate);
    of.precision(std::numeric_limits< double >::max_digits10);
    for(std::size_t i=0; i<Tlist.size();++i){
        for(Eigen::Index ri=0; ri<Tlist[i].rows(); ++ri)
            for(Eigen::Index ci=0; ci<Tlist[i].cols(); ++ci){
                of << std::setprecision(6) << Tlist[i](ri,ci);
                if(!(ri==Tlist[i].rows()-1 && ci==Tlist[i].cols()-1))
                    of << " ";
            }
        if(i!=Tlist.size()-1) of << "\n";
    }
    of.close();
}

void visual_odometry(ORB_SLAM2::System* SLAM, std::vector<Eigen::Isometry3d> &Twc_list,
    std::vector<std::size_t> &vKFFrameId, std::vector<std::string> &vstrImageFilename, std::vector<double> &vtimestamp,
     const std::string &KeyFrameDir, const std::string &MapFile)
    {
    for(std::size_t i=0; i<vstrImageFilename.size(); ++i){
        try{
            cv::Mat img;
            readImage(vstrImageFilename[i], img, cv::IMREAD_UNCHANGED);
            double tframe = vtimestamp[i];
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            SLAM->TrackMonocular(img, vtimestamp[i]);
            char msg[100];
            sprintf(msg, "\033[033;1m[VO]\033[0m Frame %ld | %ld", i+1, vstrImageFilename.size());
            std::cout << msg << std::endl;
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            double T=0;
            if(i<vstrImageFilename.size()-1)
                T = vtimestamp[i+1]-tframe;
            else if(i>0)
                T = tframe-vtimestamp[i-1];

            if(ttrack<T){
                usleep(1e6*(T-ttrack));
            }
        }catch(const std::exception &e){
            std::cout << "Visual Odometry Fatal Error: " << e.what() << std::endl;
        }
        
    }
    SLAM->Shutdown();
    std::vector<cv::Mat> vKFPose;
    SLAM->GetKeyFramePoses(vKFPose, vKFFrameId);
    for(std::size_t i=0; i<vKFPose.size(); ++i){
        Eigen::Isometry3d Twc;
        cv::cv2eigen(vKFPose[i], Twc.matrix());
        Twc_list.push_back(Twc);
    }
    std::cout << "\033[33;1mVisual Odometry Finished!\033[0m" << std::endl;
    
    SLAM->SaveKeyFrames(KeyFrameDir);
    SLAM->SaveMap(MapFile);

}

int main(int argc, char** argv){
    if(argc < 2){
        std::cout << "\033[31;1m Expected args: config \033[0m" << std::endl;
        return -1;
    }
    YAML::Node config = YAML::LoadFile(argv[1]);
    std::string root_path = config["root_path"].as<std::string>();
    checkpath(root_path);
    const YAML::Node &orb_config = config["orb"];
    const YAML::Node &io_config = config["io"];
    std::string seq_dir = root_path + io_config["seq_dir"].as<std::string>();
    std::string orb_yml = root_path + orb_config["setting"].as<std::string>();
    std::string orb_voc = root_path + orb_config["Vocabulary"].as<std::string>();
    std::string TwcFile = root_path + io_config["save_pose"].as<std::string>(); // output
    std::string keyframe_dir = root_path + io_config["keyframe_dir"].as<std::string>(); // output
    std::string mapfile = root_path + io_config["MapFile"].as<std::string>();
    bool vis = orb_config["vis"].as<bool>();
    checkpath(seq_dir);
    checkpath(keyframe_dir);
    if(file_exist(keyframe_dir)){
        std::cout << "Directory " << keyframe_dir  << " for saving existed. Try to Delete it." << std::endl;
        remove_directory(keyframe_dir.c_str());
        std::cout << "Existing Directory deleted successsfully." << std::endl;
    }
    makedir(keyframe_dir.c_str());
    check_exist(seq_dir);
    check_exist(orb_yml);
    check_exist(orb_voc);
    std::string img_dir = seq_dir + orb_config["image_dir"].as<std::string>();
    std::string timefile = seq_dir + orb_config["timestamp"].as<std::string>();
    checkpath(img_dir);
    std::vector<std::string> vImageFiles;
    std::vector<double> vTimeStamps;
    LoadTimestamp(timefile, vTimeStamps);
    LoadFiles(img_dir, vImageFiles);
    if(!(vTimeStamps.size()==vImageFiles.size())){
        char msg[100];
        sprintf(msg, "Filesize error Timestamps (%ld) != ImageFiles (%ld)", vTimeStamps.size(), vImageFiles.size());
        throw std::runtime_error(msg);
        return -1;
    }
    std::cout << "Found " << vTimeStamps.size() << " Frames of Timestamps, Laser Scans and Images." << std::endl;
    ORB_SLAM2::System* orbSLAM(new ORB_SLAM2::System(orb_voc, orb_yml, ORB_SLAM2::System::MONOCULAR, vis)); // No Interface for VO
    std::vector<Eigen::Isometry3d> vTwc;
    std::vector<std::size_t> vKFFrameId;
    vTwc.reserve(vImageFiles.size());
    visual_odometry(orbSLAM, vTwc, vKFFrameId, vImageFiles, vTimeStamps, keyframe_dir, mapfile);
    std::cout << "Total visual Frames: " << vTwc.size() << ", Key Frames: " << vKFFrameId.size() << std::endl;
    writeData(TwcFile, vTwc);

    
}