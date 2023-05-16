#pragma once
#include <fstream>
#include <iostream>
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
        return system(cmd);
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
        double data[16];
        for(int i=0; i<12; ++i){
            ifs >> data[i];
        }
        data[12] = 0.0;
        data[13] = 0.0;
        data[14] = 0.0;
        data[15] = 1.0;
        Eigen::Map<Eigen::Matrix4d, Eigen::RowMajor> mat4d(data);
        Eigen::Isometry3d mat;  //
        mat.matrix() = mat4d.matrix().transpose(); 
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
void writeSim3(const std::string &outfile, const Eigen::Isometry3d &pose, const double scale){
    std::ofstream of(outfile, std::ios::ate);
    of.precision(std::numeric_limits< double >::max_digits10);
    for(Eigen::Index ri=0; ri<3; ++ri)
        for(Eigen::Index ci=0; ci<4; ++ci){
            of << pose(ri,ci) << " ";
        }
    of << scale;
    of.close();
}

/**
 * @brief write 13 entries: 12 for 3x4 Rigid, 1 for scale
 * 
 * @param outfile File to Write (ASCII)
 * @param pose Rigid Transformation Matrix
 * @param scale Monocular Scale Factor
 */
void writeSim3(const std::string &outfile, const Eigen::Matrix4d &pose, const double scale){
    std::ofstream of(outfile, std::ios::ate);
    of.precision(std::numeric_limits< double >::max_digits10);
    for(Eigen::Index ri=0; ri<3; ++ri)
        for(Eigen::Index ci=0; ci<4; ++ci){
            of << pose(ri,ci) << " ";
        }
    of << scale;
    of.close();
}

std::tuple<Eigen::Matrix4d, double> readSim3(const std::string &file){
    std::ifstream ifs(file);
    double scale = 1.0;
    Eigen::Matrix4d mat(Eigen::Matrix4d::Identity());
    for(Eigen::Index ri=0; ri<3; ++ri)
        for(Eigen::Index ci=0; ci<4; ++ci){
            ifs >> mat(ri,ci);
        }
    ifs >> scale;
    ifs.close();
    return {mat, scale};
}

void pose2Motion(std::vector<Eigen::Isometry3d> &vAbsPose, std::vector<Eigen::Isometry3d> &vRelPose){
    for(std::size_t i = 0; i < vAbsPose.size()-1; ++i){
        Eigen::Isometry3d mTwc(vAbsPose[i+1] * vAbsPose[i].inverse());  // T(i+1) * T(i).inverse()
        vRelPose.push_back(mTwc);
    }
}