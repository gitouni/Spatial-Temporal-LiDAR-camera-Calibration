#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include "pcl/io/pcd_io.h"
#include "kitti_tools.h"
#include <opencv2/highgui/highgui.hpp>

/**
 * @brief Read point cloud into pcl::PointCloud
 * 
 * @param filename 
 * @param point_cloud 
 * @param skip read one point per "skip" points, skip=1 menas loading all points
 * @return true 
 * @return false 
 */
bool readPointCloud(const std::string filename, pcl::PointCloud<pcl::PointXYZI> &point_cloud, unsigned short skip=1)
{
    std::string suffix = filename.substr(filename.find_last_of('.') + 1);
    std::size_t cnt = 0;
    if(suffix=="pcd"){
        pcl::io::loadPCDFile(filename, point_cloud);
        return true;
    }else if(suffix=="bin"){
        std::ifstream binfile(filename.c_str(),std::ios::binary);
        if(!binfile)
        {
            throw std::runtime_error("file " + filename + " cannot open");
            return false;
        }
        else
        {
            binfile.seekg(0, std::ios::end);
            const auto file_size = binfile.tellg();
            std::size_t num_points = file_size / 4; 
            binfile.seekg(0, std::ios::beg);
            point_cloud.reserve(num_points / skip);
            for(std::size_t i = 0; i <= num_points - skip; i += skip)
            {
                float s;
                pcl::PointXYZI point;
                binfile.read((char*)&s,sizeof(float));
                point.x = s;
                binfile.read((char*)&s,sizeof(float));
                point.y = s;
                binfile.read((char*)&s,sizeof(float));
                point.z = s;
                binfile.read((char*)&s,sizeof(float));
                point.intensity = s;
                point_cloud.push_back(point);
            }
        }
        binfile.close();
        return true;
    }else{
        throw std::invalid_argument("Error read Type, got " + suffix +" ,must be .bin or .pcd");
        return false;
    }
}


/**
 * @brief Read point cloud into pcl::PointCloud
 * 
 * @param filename 
 * @param point_cloud 
 * @param skip read one point per "skip" points, skip=1 menas loading all points
 * @return true 
 * @return false 
 */
bool readPointCloud(const std::string filename, pcl::PointCloud<pcl::PointXYZ> &point_cloud, unsigned short skip=1)
{
    std::string suffix = filename.substr(filename.find_last_of('.') + 1);
    std::size_t cnt = 0;
    if(suffix=="pcd"){
        pcl::io::loadPCDFile(filename, point_cloud);
        return true;
    }else if(suffix=="bin"){
        std::ifstream binfile(filename.c_str(),std::ios::binary);
        if(!binfile)
        {
            throw std::runtime_error("file " + filename + " cannot open");
            return false;
        }
        else
        {
            binfile.seekg(0, std::ios::end);
            const auto file_size = binfile.tellg();
            std::size_t num_points = file_size / 4; 
            binfile.seekg(0, std::ios::beg);
            point_cloud.reserve(num_points / skip);
            for(std::size_t i = 0; i <= num_points - skip; i += skip)
            {
                float s;
                pcl::PointXYZ point;
                binfile.read((char*)&s,sizeof(float));
                point.x = s;
                binfile.read((char*)&s,sizeof(float));
                point.y = s;
                binfile.read((char*)&s,sizeof(float));
                point.z = s;
                binfile.read((char*)&s,sizeof(float));
                point_cloud.push_back(point);
            }
        }
        binfile.close();
        return true;
    }else{
        throw std::invalid_argument("Error read Type, got " + filename +" , its suffix must be .bin or .pcd");
        return false;
    }
}

/**
 * @brief Read Point Cloud to a vector of Eigen::Vector3d
 * 
 * @param filename 
 * @param points vector of xyz points
 * @param skip read one point per "skip" points, skip=1 menas loading all points
 * @return true 
 * @return false 
 */
bool readPointCloud(const std::string filename, std::vector<Eigen::Vector3d> &points, unsigned short skip=1)
{
    std::string suffix = filename.substr(filename.find_last_of('.') + 1);
    std::size_t cnt = 0;
    if(suffix=="pcd"){
        pcl::PointCloud<pcl::PointXYZ> point_cloud;
        pcl::io::loadPCDFile(filename, point_cloud);
        for(auto &pt:point_cloud){
            Eigen::Vector3d pt_eigen = pt.getVector3fMap().cast<double>();
            points.push_back(pt_eigen);
        }
        return true;
    }else if(suffix=="bin"){
        std::ifstream binfile(filename.c_str(),std::ios::binary);
        if(!binfile)
        {
            throw std::runtime_error("file " + filename + " cannot open");
            return false;
        }
        else
        {
            binfile.seekg(0, std::ios::end);
            const auto file_size = binfile.tellg();
            std::size_t num_points = file_size / 4; 
            binfile.seekg(0, std::ios::beg);
            points.reserve(num_points / skip);
            for(std::size_t i = 0; i <= num_points - skip; i += skip)
            {
                float s;
                Eigen::Vector3d point;
                binfile.read((char*)&s,sizeof(float));
                point(0) = s;
                binfile.read((char*)&s,sizeof(float));
                point(1) = s;
                binfile.read((char*)&s,sizeof(float));
                point(2) = s;
                binfile.read((char*)&s,sizeof(float));
                // s is intensity and won't be loaded to Vector3d
                points.push_back(point);
            }
        }
        binfile.close();
        return true;
    }else{
        throw std::invalid_argument("Error read Type, got " + filename +" , its suffix must be .bin or .pcd");
        return false;
    }
}

/**
 * @brief Read Point Cloud to a vector of Eigen::Vector4d
 * 
 * @param filename 
 * @param points vector of xyzi points
 * @param skip read one point per "skip" points, skip=1 menas loading all points
 * @return true 
 * @return false 
 */
bool readPointCloud(const std::string filename, std::vector<Eigen::Vector4d> &points, unsigned short skip=1)
{
    std::string suffix = filename.substr(filename.find_last_of('.') + 1);
    std::size_t cnt = 0;
    if(suffix=="pcd"){
        pcl::PointCloud<pcl::PointXYZI> point_cloud;
        pcl::io::loadPCDFile(filename, point_cloud);
        for(auto &pt:point_cloud){
            Eigen::Vector4d pt_eigen = pt.getVector4fMap().cast<double>();
            points.push_back(pt_eigen);
        }
        return true;
    }else if(suffix=="bin"){
        std::ifstream binfile(filename.c_str(),std::ios::binary);
        if(!binfile)
        {
            throw std::runtime_error("file " + filename + " cannot open");
            return false;
        }
        else
        {
            binfile.seekg(0, std::ios::end);
            const auto file_size = binfile.tellg();
            std::size_t num_points = file_size / 4; 
            binfile.seekg(0, std::ios::beg);
            points.reserve(num_points / skip);
            for(std::size_t i = 0; i <= num_points - skip; i += skip)
            {
                float s;
                Eigen::Vector4d point;
                binfile.read((char*)&s,sizeof(float));
                point(0) = s;
                binfile.read((char*)&s,sizeof(float));
                point(1) = s;
                binfile.read((char*)&s,sizeof(float));
                point(2) = s;
                binfile.read((char*)&s,sizeof(float));
                point(3) = s;
                points.push_back(point);
            }
        }
        binfile.close();
        return true;
    }else{
        throw std::invalid_argument("Error read Type, got " + filename +" ,its sffuix must be .bin or .pcd");
        return false;
    }
}


bool readImage(const std::string &strImageFilename, cv::Mat &img, int flag=cv::IMREAD_UNCHANGED){
    img = cv::imread(strImageFilename, flag);
    return true;
}

/**
 * @brief KITTI pose files must have 12 entries per row and no trailing delimiter at the end of the rows (space)
 * 
 * @param fs // output file stream
 * @param mat  // pose 3x4 or 4x4
 */
void writeKittiData(std::ofstream &fs, const Eigen::Isometry3d &mat, bool end){
    short cnt = 0;
    for(short i=0; i<3; ++i)
        for(short j=0; j<4; ++j){
            fs << mat(i,j);
            ++cnt;
            if(cnt!=12) fs << " ";
            else if(!end) fs << "\n";
        }
}


void LoadFiles(const std::string strDirectory,std::vector<std::string> &vstrFilenames)
{
    if(!file_exist(strDirectory)){
        throw std::runtime_error("Data Directory: "+strDirectory+" does not exsit.");
        return;
    };  // time.txt must exist
    
    listdir(vstrFilenames, strDirectory);
    for(auto &fileNames: vstrFilenames)
        fileNames = strDirectory + fileNames;
}

void LoadTimestamp(const std::string strPathTimeFile, std::vector<double> &vTimestamps){
    std::ifstream fTimes;
    if(!file_exist(strPathTimeFile)){
        throw std::runtime_error("Timestamp file: "+strPathTimeFile+" does not exsit.");
        return;
    };  // time.txt must exist
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        std::string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
}