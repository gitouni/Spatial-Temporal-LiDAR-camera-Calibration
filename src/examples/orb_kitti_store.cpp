#include "io_tools.h"
#include "orb_slam/include/System.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <thread>
#include <chrono>
#include <fstream>
#include <limits>

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
    if(file_exist(KeyFrameDir)){
        std::cout << "Directory " << KeyFrameDir  << " for saving existed. Try to Delete it." << std::endl;
        remove_directory(KeyFrameDir.c_str());
        std::cout << "Existing Directory deleted successsfully." << std::endl;
    }
    makedir(KeyFrameDir.c_str());
    SLAM->SaveKeyFrames(KeyFrameDir);
    SLAM->SaveMap(MapFile);

}

int main(int argc, char** argv){
    if(argc < 7){
        std::cout << "\033[31;1m Expected args: kitti_seq_dir orb_setting_file orb_vocabulary_file TwcFile KeyFrameDir MapFile.\033[0m" << std::endl;
        return -1;
    }
    
    std::string seq_dir = argv[1];
    std::string orb_yml = argv[2];
    std::string orb_voc = argv[3];
    std::string TwcFile = argv[4]; // output
    std::string keyframe_dir = argv[5]; // output
    std::string mapfile = argv[6];
    checkpath(seq_dir);
    checkpath(keyframe_dir);
    makedir(keyframe_dir.c_str());
    check_exist(seq_dir);
    check_exist(orb_yml);
    check_exist(orb_voc);
    std::string img_dir = seq_dir + "image_0/";
    std::string timefile = seq_dir + "times.txt";
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
    ORB_SLAM2::System* orbSLAM(new ORB_SLAM2::System(orb_voc, orb_yml, ORB_SLAM2::System::MONOCULAR, false)); // No Interface for VO
    std::vector<Eigen::Isometry3d> vTwc;
    std::vector<std::size_t> vKFFrameId;
    vTwc.reserve(vImageFiles.size());
    visual_odometry(orbSLAM, std::ref(vTwc), std::ref(vKFFrameId), std::ref(vImageFiles), std::ref(vTimeStamps), std::ref(keyframe_dir), mapfile);
    std::cout << "Total visual Frames: " << vTwc.size() << ", Key Frames: " << vKFFrameId.size() << std::endl;
    writeData(TwcFile, vTwc);

    
}