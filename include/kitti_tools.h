#pragma once
#include <fstream>
#include <string>
#include <sys/stat.h>  // F_OK
#include <unistd.h> //  access
#include <dirent.h> // makdir
#include <algorithm>
#include <limits>
#include <Eigen/Dense>

void sort_string(std::vector<std::string> &in_array)
{
	sort(in_array.begin(), in_array.end());
}


inline void checkpath(std::string &path){
    if(path[path.size()-1] != '/')
        path = path + "/";
    }   

inline bool file_exist(const std::string path){
    return access(path.c_str(), F_OK) != -1;
}

void makedir(const char* folder){
    if(access(folder,F_OK)){
        if(mkdir(folder,0755)!=0)
            printf("\033[1;33mDirectory %s created Failed with unknown error!\033[0m\n",folder);
        else
            printf("\033[1;34mDirectory %s created successfully!\033[0m\n",folder);
    }else
        printf("\033[1;35mDirectory %s not accessible or alreadly exists!\033[0m\n",folder);
}

int remove_directory(const char *path) {
    try{
        char cmd[100];
        sprintf(cmd,"rm -rf %s",path);
        system(cmd);
        return 0;
    }catch(const std::exception &e){
        std::cout << e.what() << std::endl;
        return -1;
    }
    
}

void listdir(std::vector<std::string> &filenames, const std::string &dirname){
    filenames.clear();
    DIR *dirp = nullptr;
    struct dirent *dir_entry = nullptr;
    if((dirp=opendir(dirname.c_str()))==nullptr){
        throw std::runtime_error("Cannot open directory");
    }
    while((dir_entry=readdir(dirp))!=nullptr){
        if(dir_entry->d_type!=DT_REG)
            continue;
        filenames.push_back(dir_entry->d_name);
    }
    closedir(dirp);
    sort_string(filenames);
}


void ReadPoseList(const std::string &fileName, std::vector<Eigen::Isometry3d> &vPose){
    if(!file_exist(fileName)){
        throw std::runtime_error("Cannot open file: "+ fileName);
        return;
    }
    std::ifstream ifs(fileName);
    while(ifs.peek()!=EOF){
        std::vector<double> data(12,0);
        for(auto &d:data){
            ifs >> d;
        }
        Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();  // 
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        rotation << data[0], data[1], data[2], data[4], data[5], data[6], data[8], data[9], data[10];
        translation << data[3], data[7], data[11];
        mat.rotate(rotation);
        mat.pretranslate(translation);
        vPose.push_back(mat);
    }
    ifs.close();
}

/**
 * @brief write 13 entries: 12 for 3x4 Rigid, 1 for scale
 * 
 * @param outfile File to Write (ASCII)
 * @param pose Rigid Transformation Matrix
 * @param scale Monocular Scale Factor
 */
void writePose(std::string &outfile, const Eigen::Isometry3d &pose, const double scale){
    std::ofstream of(outfile, std::ios::ate);
    of.precision(std::numeric_limits< double >::max_digits10);
    for(Eigen::Index ri=0; ri<pose.rows(); ++ri)
        for(Eigen::Index ci=0; ci<pose.cols(); ++ci){
            of << pose(ri,ci) << " ";
        }
    of << scale;
    of.close();
}

void pose2Motion(std::vector<Eigen::Isometry3d> &vAbsPose, std::vector<Eigen::Isometry3d> &vRelPose){
    for(std::size_t i = 0; i < vAbsPose.size()-1; ++i){
        Eigen::Isometry3d mTwc(vAbsPose[i+1] * vAbsPose[i].inverse());  // T(i+1) * T(i).inverse()
        vRelPose.push_back(mTwc);
    }
}