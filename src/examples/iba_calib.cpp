#include "yaml-cpp/yaml.h"
#include "io_tools.h"
#include "kitti_tools.h"
#include "pointcloud.h"
#include "IBACalib.hpp"
#include "nanoflann.hpp"
#include "orb_slam/include/System.h"
#include "orb_slam/include/KeyFrame.h"
#include <opencv2/core/eigen.hpp>
#include <limits>

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, VecVector2d>, VecVector2d, 2, std::uint32_t> KDTree2D;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, VecVector3d>, VecVector3d, 3, std::uint32_t> KDTree3D;

class IBAParams{
public:
    IBAParams(){};
public:
    const double max_pixel_dist = 1.5;
    const int kdtree2d_max_nn = 10;
    const int kdtree3d_max_nn = 30;
    const double norm_radius = 0.6;
    const int max_iba_iter = 30;
    const bool verborse = true;
};



void FindProjectCorrespondences(const VecVector3d &points, const ORB_SLAM2::KeyFrame* KeyFrame,
    const int kdtree_nn, const double max_corr_dist, std::vector<std::pair<std::uint32_t, std::uint32_t>> &corrset)
{
    VecVector2d vKeyUnEigen;
    vKeyUnEigen.reserve(KeyFrame->mvKeysUn.size());
    for (auto &KeyUn:KeyFrame->mvKeysUn)
    {
        vKeyUnEigen.push_back(Eigen::Vector2d(KeyUn.pt.x, KeyUn.pt.y));
    }   
    double fx = KeyFrame->fx, fy = KeyFrame->fy, cx = KeyFrame->cx, cy = KeyFrame->cy;
    double H = KeyFrame->mnMaxY, W = KeyFrame->mnMaxX;
    VecVector2d ProjectPC;
    std::vector<std::uint32_t> ProjectIndex;
    ProjectPC.reserve(vKeyUnEigen.size());  // Actually, much larger than its size
    for (std::uint32_t i = 0; i < (std::uint32_t)points.size(); ++i)
    {
        auto &point = points[i];
        if(point.z() > 0){
            double u = (fx * point.x() + cx * point.z())/point.z();
            double v = (fx * point.y() + cy * point.z())/point.z();
            if (0 <= u && u < W && 0 <= v && v < H){
                ProjectPC.push_back(Eigen::Vector2d(u, v));
                ProjectIndex.push_back(i);
            }
                
        }
    }
    std::unique_ptr<KDTree2D> kdtree(new KDTree2D(2, ProjectPC, {kdtree_nn}));
    for(std::uint32_t i = 0; i < vKeyUnEigen.size(); ++i)
    {
        std::vector<std::uint32_t> query_indices(1);
        std::vector<double> sq_dist(1, std::numeric_limits<double>::max());
        int num_res = kdtree->knnSearch(vKeyUnEigen[i].data(), 1, query_indices.data(), sq_dist.data());
        if(num_res > 0 && sq_dist[0] <= max_corr_dist)
            corrset.push_back(std::make_pair(i, ProjectIndex[query_indices[0]]));
    }
}

std::tuple<VecVector3d,std::vector<std::uint32_t>> ComputeLocalNormal(const VecVector3d &points, std::vector<std::uint32_t> indices,
    const double radius, const int max_nn, const int max_pts)
{
    std::unique_ptr<KDTree3D> kdtree(new KDTree3D(3, points, {max_nn}));
    VecVector3d normals;
    std::vector<std::uint32_t> valid_indices;
    normals.reserve(points.size());
    for (std::size_t i = 0 ; i < indices.size(); ++ i){
        auto idx = indices[i];
        std::vector<std::uint32_t> query_indices(max_pts);
        std::vector<double> sq_dist(max_pts, std::numeric_limits<double>::max());
        int k = kdtree->knnSearch(points[idx].data(), 1, query_indices.data(), sq_dist.data());
        k = std::distance(sq_dist.begin(),
                      std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                       radius * radius)); // iterator difference between start and last valid index
        query_indices.resize(k);
        sq_dist.resize(k);
        if(k>=3)
        {
            Eigen::Matrix3d covariance = ComputeCovariance(points, query_indices);
            Eigen::Vector3d normal = FastEigen3x3(covariance);
            normals.push_back(normal); // self-to-self distance is minimum
            valid_indices.push_back(i);
        }
        
    }
    return {normals, valid_indices};
}

void GetCorrespondenceNormal(const std::vector<std::string> &PointCloudFiles, const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames)
{

}


int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Require config_yaml arg" << std::endl;
        exit(0);
    }
    std::string config_file(argv[1]);
    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node io_config = config["io"];
    YAML::Node orb_config = config["orb"];
    YAML::Node runtime_config = config["runtime"];
    std::string base_dir = io_config["BaseDir"].as<std::string>();
    checkpath(base_dir);
    std::vector<Eigen::Isometry3d> vTwc, vTwl, vTwlraw;
    // IO Config
    const std::string TwcFile = base_dir + io_config["VOFile"].as<std::string>();
    const std::string TwlFile = base_dir + io_config["LOFile"].as<std::string>();
    const std::string resFile = base_dir + io_config["ResFile"].as<std::string>();
    const std::string KyeFrameIdFile = base_dir + io_config["VOIdFile"].as<std::string>();
    const std::string initSim3File = base_dir + io_config["init_sim3"].as<std::string>();
    // ORB Config
    const std::string ORBVocFile = orb_config["Vocabulary"].as<std::string>();
    const std::string ORBCfgFile = orb_config["Config"].as<std::string>();
    std::string ORBKeyFrameDir = orb_config["KeyFrameDir"].as<std::string>();
    checkpath(ORBKeyFrameDir);
    const std::string ORBMapFile = orb_config["MapFile"].as<std::string>();
    // runtime config
    const bool verborse = runtime_config["verborse"].as<bool>();
    const double max_pixel_dist = runtime_config["max_pixel_dist"].as<double>();
    const double norm_radius = runtime_config["norm_radius"].as<double>();
    const int max_iba_iter = runtime_config["max_iba_iter"].as<int>();
    ReadPoseList(TwcFile, vTwc);
    ReadPoseList(TwlFile, vTwlraw);
    YAML::Node FrameIdCfg = YAML::LoadFile(KyeFrameIdFile);
    std::vector<int> vKFFrameId = FrameIdCfg["mnFrameId"].as<std::vector<int>>();
    for(auto &KFId:vKFFrameId)
        vTwl.push_back(vTwlraw[KFId]);
    vTwlraw.clear();  // Release Cached Memory

    ORB_SLAM2::System orb_slam(ORBVocFile, ORBCfgFile, ORB_SLAM2::System::MONOCULAR, false);
    orb_slam.Shutdown(); // Do not need any ORB running threads
    orb_slam.RestoreSystemFromFile(ORBKeyFrameDir, ORBMapFile);
    std::vector<ORB_SLAM2::KeyFrame*> vKeyFrame = orb_slam.GetAllKeyFrames(true);
    Eigen::Matrix4d rigid;
    double scale;
    std::tie(rigid, scale) = readSim3(initSim3File);
    


}