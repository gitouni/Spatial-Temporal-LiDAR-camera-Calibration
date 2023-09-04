#include "floam/include/floamClass.h"
#include "pcl/io/pcd_io.h"
#include "kitti_tools.h"
#include <vector>
#include <string>
#include <iostream>
#include "argparse.hpp"

template <typename PointType>
bool readPointCloud(pcl::PointCloud<PointType> &point_cloud, const std::string filename);

void writeKittiData(std::ofstream &fs, Eigen::Isometry3d &mat, bool end=false);

int main(int argc, char **argv)
{
    argparse::ArgumentParser parser("floam_kitti");
    parser.add_argument("velo_dir").help("directory of velodyne pcd files").required();
    parser.add_argument("output").help("file to write output poses").required();
    parser.parse_args(argc, argv);
    std::string velo_dir(parser.get<std::string>("velo_dir")), res_file(parser.get<std::string>("output"));
    checkpath(velo_dir);
    if(!file_exist(velo_dir)){
        throw std::runtime_error("Velodyne Directory: "+velo_dir+" does not exsit.");
    };  // velodyne/ must exist
    std::vector<std::string> velo_files;
    listdir(velo_files,velo_dir);
    std::cout << "Found " << velo_files.size() << " velodyne files." << std::endl;
    std::ofstream res_stream(res_file.c_str(), std::ios::ate);  // write mode | ASCII mode
    res_stream << std::setprecision(6);  // fixed precision;
    FLOAM::System SLAM(FLOAM::HDL_64);
    std::cout << "Create FLOAM SLAM successfully." << std::endl;
    for(std::size_t i=0; i<velo_files.size(); ++i){
        try{
            pcl::PointCloud<FLOAM::PointType>::Ptr laserCloudPt (new pcl::PointCloud<FLOAM::PointType>);
            bool isok = readPointCloud<FLOAM::PointType>(*laserCloudPt, velo_dir + velo_files[i]);
            if(!isok){
                throw std::runtime_error("Fatal error in reading " + velo_files[i] + " , exit.");
                return -1;
            }
            char msg[60];
            sprintf(msg, "Frame %06ld | %06ld: %ld points", i+1, velo_files.size(), laserCloudPt->size());
            std::cout << msg << std::endl;
            Eigen::Isometry3d pose = SLAM.Track(laserCloudPt);
            writeKittiData(res_stream, pose, i==velo_files.size()-1);
            
        }catch(const std::exception &e){
            std::cout << e.what() << std::endl;
            res_stream.close();
            return -1;
        }
        
    }
    res_stream.close();
    return 0;
}

template <typename PointType>
bool readPointCloud(pcl::PointCloud<PointType> &point_cloud, const std::string filename)
{
    std::string suffix = filename.substr(filename.find_last_of('.') + 1);
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
            while(1)
            {
            float s;
            PointType point;
            binfile.read((char*)&s,sizeof(float));
            if(binfile.eof()) break;
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
 * @brief KITTI pose files must have 12 entries per row and no trailing delimiter at the end of the rows (space)
 * 
 * @param fs // output file stream
 * @param mat  // pose 3x4 or 4x4
 */
void writeKittiData(std::ofstream &fs, Eigen::Isometry3d &mat, bool end){
    short cnt = 0;
    for(short i=0; i<3; ++i)
        for(short j=0; j<4; ++j){
            fs << mat(i,j);
            ++cnt;
            if(cnt!=12) fs << " ";
            else if(!end) fs << "\n";
        }
}