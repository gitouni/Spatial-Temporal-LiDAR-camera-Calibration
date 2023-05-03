#include "io_tools.h"
#include "orb_slam/include/System.h"
#include "floam/include/floamClass.h"
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

void lidar_odometry(std::shared_ptr<FLOAM::System> SLAM, std::vector<Eigen::Isometry3d> &Twl_list,
    std::vector<std::string> &vstrLiDARFilename, std::vector<double> &vtimestamp){
    for(std::size_t i=0; i<vstrLiDARFilename.size(); ++i){
        try{
            double tframe = vtimestamp[i];
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            pcl::PointCloud<FLOAM::PointType>::Ptr laserCloudIn(new pcl::PointCloud<FLOAM::PointType>);
            readPointCloud(vstrLiDARFilename[i], *laserCloudIn);
            Eigen::Isometry3d Twl = SLAM->Track(laserCloudIn);
            Twl_list.push_back(Twl);
            char msg[100];
            sprintf(msg, "\033[032;1m[LO]\033[0m Frame %ld | %ld", i+1, vstrLiDARFilename.size());
            std::cout << msg << std::endl;
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            double T = 0;
            if(i<vstrLiDARFilename.size()-1)
                T = vtimestamp[i+1]-tframe;
            else if(i>0)
                T = tframe-vtimestamp[i-1];

            if(ttrack<T){
                std::chrono::milliseconds dura((unsigned int)(1000*(T-ttrack)));
                std::this_thread::sleep_for(dura);
            }
        }catch(const std::exception &e){
            std::cout << "LiDAR Odometry Fatal Error: " << e.what() << std::endl;
        }
        
    
    }
    std::cout << "\033[32;1mLiDAR Odometry Finished!\033[0m" << std::endl;
}   

void visual_odometry(std::shared_ptr<ORB_SLAM2::System> SLAM, std::vector<Eigen::Isometry3d> &Twc_list,
    std::vector<std::size_t> &vKFFrameId, std::vector<std::string> &vstrImageFilename, std::vector<double> &vtimestamp, std::string &KeyFrameDir)
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
                std::chrono::milliseconds dura((unsigned int)(1000*(T-ttrack)));
                std::this_thread::sleep_for(dura);
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
    checkpath(KeyFrameDir);
    if(file_exist(KeyFrameDir)){
        std::cout << "Directory for saving existed. Try to Delete it." << std::endl;
        remove_directory(KeyFrameDir.c_str());
        std::cout << "Existing Directory deleted successsfully." << std::endl;
    }
    makedir(KeyFrameDir.c_str());
    SLAM->SaveKeyFrames(KeyFrameDir+"KeyFrames");
    SLAM->SaveMap(KeyFrameDir+"Map.xml");

}

int main(int argc, char** argv){
    if(argc < 4){
        std::cout << "\033[31;1m Got " << argc-1 << " Parameters, expect 3.\033[0m" << std::endl;
        throw std::invalid_argument("Expected args: kitti_seq_dir orb_setting_file orb_vocabulary_file [TwcFile] [TwlFile] [KeyFrameDir]");
        return -1;
    }
    
    std::string seq_dir = argv[1];
    std::string orb_yml = argv[2];
    std::string orb_voc = argv[3];
    std::string save_dir = "../tmp/";
    
    std::string TwcFile, TwlFile;
    if(argc >= 6){
        TwcFile = argv[4];
        TwlFile = argv[5];
    }else{
        TwcFile = "../Twc.txt";
        TwlFile = "../Twl.txt";
    }
    if(argc >= 7)
        save_dir.assign(argv[6]);
    checkpath(seq_dir);
    check_exist(seq_dir);
    check_exist(orb_yml);
    check_exist(orb_voc);
    std::string velo_dir = seq_dir + "velodyne/";
    std::string img_dir = seq_dir + "image_0/";
    std::string timefile = seq_dir + "times.txt";
    std::vector<std::string> vLiDARFiles, vImageFiles;
    std::vector<double> vTimeStamps;
    LoadTimestamp(timefile, vTimeStamps);
    LoadFiles(velo_dir, vLiDARFiles);
    LoadFiles(img_dir, vImageFiles);
    if(!(vTimeStamps.size()==vLiDARFiles.size() && vLiDARFiles.size()==vImageFiles.size())){
        char msg[100];
        sprintf(msg, "Filesize error !(Timestamps (%ld) == LiDARFiles (%ld) == ImageFiles (%ld))", vTimeStamps.size(), vLiDARFiles.size(), vImageFiles.size());
        throw std::runtime_error(msg);
        return -1;
    }
    std::cout << "Found " << vTimeStamps.size() << " Frames of Timestamps, Laser Scans and Images." << std::endl;
    std::shared_ptr<FLOAM::System> floamSLAM(new FLOAM::System(FLOAM::HDL_64));
    std::shared_ptr<ORB_SLAM2::System> orbSLAM(new ORB_SLAM2::System(orb_voc, orb_yml, ORB_SLAM2::System::MONOCULAR, false)); // No Interface for VO
    std::vector<Eigen::Isometry3d> vTwc, vTwl;
    std::vector<std::size_t> vKFFrameId;
    vTwc.reserve(vImageFiles.size());
    vTwl.reserve(vLiDARFiles.size());
    std::thread *visual_thread(new thread(visual_odometry, orbSLAM, std::ref(vTwc), std::ref(vKFFrameId), std::ref(vImageFiles), std::ref(vTimeStamps), std::ref(save_dir)));
    std::thread *lidar_thread(new thread(lidar_odometry, floamSLAM, std::ref(vTwl), std::ref(vLiDARFiles), std::ref(vTimeStamps)));
    lidar_thread->join();
    visual_thread->join();
    delete lidar_thread;
    delete visual_thread;
    std::cout << "Total visual Frames: " << vTwc.size() << std::endl;
    std::cout << "Total LiDAR Frames: " << vTwl.size() << std::endl;
    std::vector<Eigen::Isometry3d> vTwlSyn;
    Eigen::Isometry3d invTl0 = vTwl[vKFFrameId[0]].inverse();
    for(auto &KFId:vKFFrameId){
        vTwlSyn.push_back(invTl0 * vTwl[KFId]);
    }
    writeData(TwcFile, vTwc);
    writeData(TwlFile, vTwlSyn);
    // Eigen::Matrix3d RCL;
    // Eigen::Vector3d tCL;
    // double s;
    // std::tie(RCL,tCL,s) = HECalib(vTwc, vTwl);
    // std::cout << "Rotation: \n" << RCL << std::endl;
    // std::cout << "Translation: \n" << tCL << std::endl;
    // std::cout << "s :" << s << std::endl;
    
}