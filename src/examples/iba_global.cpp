#include "KDTreeVectorOfVectorsAdaptor.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o_tools.h"
#include "io_tools.h"
#include "kitti_tools.h"
#include "pointcloud.h"
#include <functional>
#include <limits>
#include <yaml-cpp/yaml.h>
#include "orb_slam/include/System.h"
#include "orb_slam/include/KeyFrame.h"
#include <opencv2/core/eigen.hpp>
#include "Nomad/nomad.hpp"
#include "Cache/CacheBase.hpp"
#include <unordered_map>

typedef std::uint32_t IndexType; // other types cause error, why?
typedef std::vector<IndexType> VecIndex;
typedef std::pair<IndexType, IndexType> CorrType;
typedef std::vector<CorrType> CorrSet;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector2d, double, 2, nanoflann::metric_L2_Simple, IndexType> KDTree2D;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector3d, double, 3, nanoflann::metric_L2_Simple, IndexType> KDTree3D;


class IBAGlobalParams{
public:
    IBAGlobalParams(){};
public:
    double max_pixel_dist = 1.5;
    int kdtree2d_max_leaf_size = 10;
    int kdtree3d_max_leaf_size = 30;
    int num_best_covis = 1;
    int min_covis_weight = 150;
    double corr_3d_2d_threshold = 40.;
    double corr_3d_3d_threshold = 5.;
    double he_threshold = 0.05;
    int norm_max_pts = 30;
    int norm_min_pts = 5;
    double norm_radius = 0.6;
    double norm_reg_threshold = 0.04;
    double min_diff_dist = 0.01;
    std::vector<double> err_weight;
    std::vector<double> lb;
    std::vector<double> ub;
    int PointCloudskip = 1;
    bool PointCloudOnlyPositiveX = false;
    int max_bbeval = 200;
    double valid_rate = 0.5;
    bool verborse = true;
    bool use_plane = true;
};


void FindProjectCorrespondences(const VecVector3d &points, const ORB_SLAM2::KeyFrame* pKF,
    const int leaf_size, const double max_corr_dist, CorrSet &corrset)
{
    VecVector2d vKeyUnEigen;
    vKeyUnEigen.reserve(pKF->mvKeysUn.size());
    for (auto &KeyUn:pKF->mvKeysUn)
    {
        vKeyUnEigen.push_back(Eigen::Vector2d(KeyUn.pt.x, KeyUn.pt.y));
    }   
    const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
    const double H = pKF->mnMaxY, W = pKF->mnMaxX;
    VecVector2d ProjectPC;
    VecIndex ProjectIndex;
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        auto &point = points[i];
        if(point.z() > 0){
            double u = (fx * point.x() + cx * point.z())/point.z();
            double v = (fx * point.y() + cy * point.z())/point.z();
            if (0 <= u && u < W && 0 <= v && v < H)
            {
                ProjectPC.push_back(Eigen::Vector2d(u, v));
                ProjectIndex.push_back(i);
            }
                
        }
    }
    if(ProjectPC.size() == 0)
        return;
    std::unique_ptr<KDTree2D> kdtree (new KDTree2D(2, ProjectPC, leaf_size));
    for(IndexType i = 0; i < vKeyUnEigen.size(); ++i)
    {
        const int num_closest = 1;
        std::vector<IndexType> indices(num_closest);
        std::vector<double> sq_dist(num_closest, std::numeric_limits<double>::max());
        nanoflann::KNNResultSet<double, IndexType> resultSet(num_closest);
        resultSet.init(indices.data(), sq_dist.data());
        kdtree->index->findNeighbors(resultSet, vKeyUnEigen[i].data(), nanoflann::SearchParameters());
        if(resultSet.size() > 0 && sq_dist[0] <= max_corr_dist * max_corr_dist)
            corrset.push_back(std::make_pair(i, ProjectIndex[indices[0]]));
    }
}

/**
 * @brief  * @brief Return Nearest Dist between query point and PointCloud 
 * Use Point-Plane Dist first, and degenerate to Point-Point Dist if the last failed
 * 
 * @param kdtree 
 * @param PointCloud 
 * @param query_pt 
 * @param norm_max_pts 
 * @param norm_radius 
 * @param norm_reg_threshold 
 * @param min_diff_dist 
 * @return std::tuple<bool, double> 
 */
std::tuple<bool, double> ComputeAlignmentDist(const KDTree3D* kdtree, const VecVector3d &PointCloud, const Eigen::Vector3d &query_pt,
    const int &norm_max_pts, const int &norm_min_pts, const double &norm_radius,
    const double &norm_reg_threshold, const double &min_diff_dist, const bool use_plane=true)
{
    // find nearest point of Laser Scan to query_pt
    VecIndex nn_idx(1);
    std::vector<double> nn_sq_dist(1);
    nanoflann::KNNResultSet<double, IndexType> nnResSet(1);
    nnResSet.init(nn_idx.data(), nn_sq_dist.data());
    kdtree->index->findNeighbors(nnResSet, query_pt.data(), nanoflann::SearchParameters());
    const Eigen::Vector3d &nn_pt = PointCloud[nn_idx[0]];
    double pt2pt_dist = (nn_pt - query_pt).norm();
    if(!use_plane)
        return {false, pt2pt_dist};
    VecIndex indices(norm_max_pts);
    std::vector<double> sq_dist(norm_max_pts);
    nanoflann::KNNResultSet<double, IndexType> resultSet(norm_max_pts);
    resultSet.init(indices.data(), sq_dist.data());
    kdtree->index->findNeighbors(resultSet, nn_pt.data(), nanoflann::SearchParameters());  // use query_pt to find a plane
    std::size_t k = resultSet.size();
    k = std::distance(sq_dist.begin(),
                std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                norm_radius * norm_radius)); // iterator difference between start and last valid index
    indices.resize(k);
    sq_dist.resize(k);
    if(sq_dist[k-1] < min_diff_dist * min_diff_dist)
        return {false, pt2pt_dist};
    if(k < norm_min_pts)
        return {false, pt2pt_dist};
    Eigen::Matrix3d covariance = ComputeCovariance(PointCloud, indices);
    Eigen::Vector3d normal;
    std::tie(normal, std::ignore) = FastEigen3x3_EV(covariance);
    normal.normalize();
    double reg_err = 0;
    for(auto const &idx:indices)
        reg_err += std::abs((PointCloud[idx] - nn_pt).dot(normal));
    if(reg_err / (k - 1) > norm_reg_threshold)   // not include indices[0]
        return {false, pt2pt_dist};
    else
    {
        double max_dist = 0;
        double pt2pl_dist = std::abs((nn_pt - query_pt).dot(normal));
        return {true, pt2pl_dist};
    }
    
}

/**
 * @brief Compute BA Error and HE Constraint
 * 
 * @param xvec 
 * @param PointClouds 
 * @param vTwl 
 * @param KeyFrames 
 * @param iba_params 
 * @param multiprocessing 
 * @return std::tuple<double, double, ,double, int, int> f1, f2, C, valid_edge_cnt, edge_cnt
 */
std::tuple<double, double, double, int, int> BAError(
 const double* xvec, const std::vector<VecVector3d> &PointClouds, const std::vector<std::unique_ptr<KDTree3D>> &KdTrees,
 const std::vector<Eigen::Isometry3d> &vTwl, const std::unordered_map<int,int> &KFIdMap,
 const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames, const IBAGlobalParams &iba_params,
 const bool &multiprocessing=false, const bool &verborse=false)
{
    double corr_3d_2d_err = 0; // total error
    double corr_3d_3d_err = 0;
    // std::vector<double> corr_3d_3d_errlist;
    double Cval = 0; // total HE constraint
    double Ccnt = 0;
    int cnt_3d_2d = 0;
    int valid_cnt_3d_2d = 0;
    int valid_pl_3d_3d = 0;
    int valid_pt_3d_3d = 0;
    int cnt_3d_3d = 0;
    int valid_cnt_3d_3d = 0;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double scale;
    std::tie(rotation, translation, scale) = Sim3Exp<double>(xvec);
    Eigen::Isometry3d Tcl(Eigen::Isometry3d::Identity());
    Tcl.rotate(rotation);
    Tcl.pretranslate(translation);
    #pragma omp parallel for if(multiprocessing)
    for(std::size_t Fi = 0; Fi < PointClouds.size(); ++Fi)
    {
        const VecVector3d &PL = PointClouds[Fi]; // Point Cloud in Lidar coord
        VecVector3d PC; // Point Cloud in camera coord
        const ORB_SLAM2::KeyFrame* pKF = KeyFrames[Fi];  
        TransformPointCloud(PL, PC, Tcl);    
        CorrSet corrset; // pair of (idx of image points, idx of pointcloud points)
        /* Find 3D-2D Correspondence */
        FindProjectCorrespondences(PC, pKF, iba_params.kdtree2d_max_leaf_size, iba_params.max_pixel_dist, corrset);
        if(corrset.size() < 30)  // too few correspondences
            continue;
        const cv::Mat InvRefCVPose = pKF->GetPoseInverseSafe();
        Eigen::Matrix4d TcwRS;  // Tcw with real size
        cv::cv2eigen(pKF->GetPoseSafe(), TcwRS);
        TcwRS.topRightCorner(3, 1) *= scale;
        /* Compute 3D-3D Correspondences through scale */
        std::unordered_map<int, ORB_SLAM2::MapPoint*> mapKpt2Mpt;
        mapKpt2Mpt.reserve(pKF->mmapMpt2Kpt.size());
        for(auto const& [key, val] : pKF->mmapMpt2Kpt)
            mapKpt2Mpt[val] = key;
        if(iba_params.err_weight[1] <= 1e-10)
        {
            corr_3d_3d_err = 0;
            cnt_3d_3d++;
            valid_cnt_3d_3d++;
            // corr_3d_3d_errlist.push_back(0);
        }
        else
        {
            for(auto const &[point2d_idx, point3d_idx]:corrset)
            {
                if(mapKpt2Mpt.count(point2d_idx) == 0)
                    continue;
                VecIndex indices_cl(1);
                std::vector<double> sq_dist_cl(1);
                nanoflann::KNNResultSet<double, IndexType> resultSet_cl(1);
                resultSet_cl.init(indices_cl.data(), sq_dist_cl.data());
                Eigen::Vector3d MapRefPose;
                cv::cv2eigen(mapKpt2Mpt[point2d_idx]->GetWorldPos() * scale, MapRefPose);  // sPw
                MapRefPose = TcwRS.topLeftCorner(3,3) * MapRefPose + TcwRS.topRightCorner(3,1); // Pci
                MapRefPose = Tcl.inverse() * MapRefPose;  // Pli
                bool is_planefit;
                double dist;
                std::tie(is_planefit, dist) = ComputeAlignmentDist(KdTrees[Fi].get(), PL, MapRefPose, iba_params.norm_max_pts,
                 iba_params.norm_min_pts,iba_params.norm_radius, iba_params.norm_reg_threshold, iba_params.min_diff_dist, iba_params.use_plane);
                #pragma omp critical
                {
                    if(dist < iba_params.corr_3d_3d_threshold)
                    {
                        corr_3d_3d_err += dist;
                        valid_cnt_3d_3d++;
                        if(is_planefit)
                            valid_pl_3d_3d++;
                        else
                            valid_pt_3d_3d++;
                    }
                    cnt_3d_3d++;
                }
            }
        }
        
        std::vector<ORB_SLAM2::KeyFrame*> pCovisKFs;
        if(iba_params.num_best_covis > 0)
            pCovisKFs = pKF->GetBestCovisibilityKeyFramesSafe(iba_params.num_best_covis);
        else
            pCovisKFs = pKF->GetCovisiblesByWeightSafe(iba_params.min_covis_weight);  
        std::vector<std::unordered_map<int, int>> KptMapList; // Keypoint-Keypoint Corr
        std::vector<Eigen::Matrix4d> relCVPoseList; // relCVPose From Reference to Covisible KeyFrames
        KptMapList.reserve(pCovisKFs.size());
        relCVPoseList.reserve(pCovisKFs.size());
        if(Fi < KeyFrames.size() - 1)
        {
            Eigen::Matrix4d Tc;
            cv::cv2eigen(KeyFrames[Fi+1]->GetPose() * InvRefCVPose, Tc); // 
            Tc.topRightCorner(3, 1) *= scale;
            Eigen::Isometry3d Tl = vTwl[Fi+1].inverse() * vTwl[Fi];
            Eigen::Matrix4d C1 = (Tcl * Tl).matrix();
            Eigen::Matrix4d C2 = Tc * Tcl.matrix();
            auto c1_log = SE3Log(C1.topLeftCorner(3, 3), C1.topRightCorner(3, 1));
            auto c2_log = SE3Log(C2.topLeftCorner(3, 3), C2.topRightCorner(3, 1));
            Cval += (c1_log - c2_log).norm();
            Ccnt++;
        }
        for(auto pKFConv:pCovisKFs)
        {
            auto KptMap = pKF->GetUordMatchedKptIds(pKFConv);
            cv::Mat relCVPose = pKFConv->GetPose() * InvRefCVPose;  // Transfer from c1 coordinate to c2 coordinate
            Eigen::Matrix4d relCVPoseEigen;
            cv::cv2eigen(relCVPose, relCVPoseEigen);
            relCVPoseEigen.topRightCorner(3, 1) *= scale; // noalias
            // KptMap and relCVPoseEigen has been cleared.
            KptMapList.push_back(std::move(KptMap));
            relCVPoseList.push_back(std::move(relCVPoseEigen));
            // Hnadeye Constraint

        }
        
        for(auto &corr:corrset)
        {
            const int point2d_idx = corr.first;  // KeyPoint Idx matched with PointCloud
            const int point3d_idx = corr.second; // Point Idx matched with KeyPoints
            double u0 = pKF->mvKeysUn[point2d_idx].pt.x;
            double v0 = pKF->mvKeysUn[point2d_idx].pt.y;
            Eigen::Vector3d p0 = PC[point3d_idx]; // camera coord
            const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
            const double H = pKF->mnMaxY, W = pKF->mnMaxX;
            // transform 3d point back to LiDAR coordinate
            for(std::size_t pKFConvi = 0; pKFConvi < pCovisKFs.size(); ++pKFConvi){
                auto pKFConv = pCovisKFs[pKFConvi];
                // Skip if Cannot Find this 2d-3d matching map in Keypoint-to-Keypoint matching map
                if(KptMapList[pKFConvi].count(point2d_idx) == 0)
                    continue;
                const int covis_idx = KptMapList[pKFConvi][point2d_idx];  // corresponding KeyPoints idx in a covisible KeyFrame
                double u1 = pKFConv->mvKeysUn[covis_idx].pt.x;
                double v1 = pKFConv->mvKeysUn[covis_idx].pt.y;
                Eigen::Matrix4d relCVPose = relCVPoseList[pKFConvi];  // Twc2 * inv(Twc1)
                Eigen::Vector3d p1 = relCVPose.topLeftCorner(3, 3) * p0 + relCVPose.topRightCorner(3, 1); // transform to covisible Keyframe coord
                double obs_u1 = fx * p1[0]/p1[2] + cx;
                double obs_v1 = fy * p1[1]/p1[2] + cy;
                if(!(obs_u1 >= 0 && obs_u1 < W && obs_v1 >=0 && obs_v1 < H))
                    continue;
                // printf("u0: %lf, v0: %lf, u1:%lf, v1:%lf, obs_u1:%lf, obs_v1:%lf\n", u0, v0, u1, v1, obs_u1, obs_v1);
                double err = (obs_u1 - u1) * (obs_u1 - u1) + (obs_v1 - v1) * (obs_v1 - v1);
                double dist = sqrt(err);
                #pragma omp critical
                {
                    if(dist < iba_params.corr_3d_2d_threshold)
                    {
                        corr_3d_2d_err += dist;
                        valid_cnt_3d_2d++;
                    }
                    cnt_3d_2d++;
                }
            }
        }
    }
    if(valid_cnt_3d_2d == 0 && iba_params.err_weight[0] > 1e-10)
        corr_3d_2d_err = std::numeric_limits<double>::max();
    else
        corr_3d_2d_err /= valid_cnt_3d_2d;
    if(valid_cnt_3d_3d == 0 && iba_params.err_weight[1] > 1e-10)
        corr_3d_3d_err = std::numeric_limits<double>::max();
    else
        corr_3d_3d_err /= valid_cnt_3d_3d;
    Cval /= Ccnt;
    // auto logger = LogEdges(corr_3d_3d_errlist);
    // print_map("corr3d_3d:",logger);
    if(verborse)
        std::printf("plane: %d, point: %d 3d-2d: %d\n", valid_pl_3d_3d, valid_pt_3d_3d, valid_cnt_3d_2d);
    return {corr_3d_2d_err, corr_3d_3d_err, Cval, valid_cnt_3d_2d, cnt_3d_2d};
}

class BALoss: public NOMAD::Evaluator
{
public:
    BALoss(const std::shared_ptr<NOMAD::EvalParameters>& evalParams, const NOMAD::EvalType & evalType,
    const std::vector<Eigen::Isometry3d>* _PointCloudPoses, const std::vector<VecVector3d>* _PointClouds,
    const std::unordered_map<int,int>* _KFIdMap, const std::vector<ORB_SLAM2::KeyFrame*>* _KeyFrames, 
    const IBAGlobalParams* _iba_params,
    const std::vector<std::pair<std::string, std::vector<double>> >& SpecialPoints):
        NOMAD::Evaluator(evalParams, evalType)
        {
            PointCloudPoses = _PointCloudPoses;
            PointClouds = _PointClouds;
            KeyFrames = _KeyFrames;
            KFIdMap = _KFIdMap;
            iba_params = _iba_params;
            std::printf("Building %ld KdTrees for PointClouds.\n", PointClouds->size());
            kdtree_list.resize(PointClouds->size());
            #pragma omp parallel for schedule(static)
            for(std::size_t i = 0; i < PointClouds->size(); ++ i)
            {
                kdtree_list[i] = std::make_unique<KDTree3D>(3, PointClouds->at(i), iba_params->kdtree3d_max_leaf_size);   
            }
            std::printf("Evaluating %ld special points.\n", SpecialPoints.size());
            for(auto &x:SpecialPoints){
                double f1val, f2val ,Cval;
                int valid_edge_cnt, edge_cnt;
                std::tie(f1val, f2val, Cval, valid_edge_cnt, edge_cnt) = BAError(x.second.data(), *PointClouds, kdtree_list, *PointCloudPoses, *KFIdMap, *KeyFrames, *iba_params, true, true);
                std::printf("Special Point %s : f1val: %lf, f2val:%lf, Cval:%lf, edge: %d, valid: %d\n", x.first.c_str(), f1val, f2val, Cval, edge_cnt, valid_edge_cnt);
            }
        }
    ~BALoss(){}
    bool eval_x(NOMAD::EvalPoint &x, const NOMAD::Double &hMax, bool &countEval) const override
    {
        double f1val = 0, f2val = 0, Cval = 0; // total error
        int edge_cnt = 0;
        int valid_edge_cnt = 0;
        double xvec[7];
        for(int i = 0; i < 7; ++i)
            xvec[i] = x[i].todouble();
        std::tie(f1val, f2val, Cval, valid_edge_cnt, edge_cnt) = BAError(xvec, *PointClouds, kdtree_list, *PointCloudPoses, *KFIdMap, *KeyFrames, *iba_params);
        NOMAD::Double f = f1val * iba_params->err_weight[0] + f2val * iba_params->err_weight[1];
        NOMAD::Double C1 = Cval - iba_params->he_threshold, C2 = -Cval - iba_params->he_threshold;  // |Cval| <= he_threshold
        NOMAD::Double C3 = iba_params->valid_rate - static_cast<double>(valid_edge_cnt) / (edge_cnt+1);
        std::string bbo = f.tostring();
        bbo += " " + C1.tostring();
        bbo += " " + C2.tostring();
        bbo += " " + C3.tostring();
        x.setBBO(bbo);
        countEval = true;
        return true;
    }

private:
    const std::vector<VecVector3d>* PointClouds;
    const std::vector<ORB_SLAM2::KeyFrame*>* KeyFrames;
    const std::unordered_map<int,int>* KFIdMap;
    const std::vector<Eigen::Isometry3d>* PointCloudPoses;
    const IBAGlobalParams* iba_params;
    std::vector<std::unique_ptr<KDTree3D>> kdtree_list;
};

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Require config_yaml arg" << std::endl;
        exit(0);
    }
    std::string config_file(argv[1]);
    YAML::Node config = YAML::LoadFile(config_file);
    const YAML::Node &io_config = config["io"];
    const YAML::Node &orb_config = config["orb"];
    const YAML::Node &runtime_config = config["runtime"];
    std::string base_dir = io_config["BaseDir"].as<std::string>();
    std::string pointcloud_dir = io_config["PointCloudDir"].as<std::string>();
    checkpath(base_dir);
    checkpath(pointcloud_dir);
    std::vector<std::string> RawPointCloudFiles, PointCloudFiles;
    std::vector<ORB_SLAM2::KeyFrame*> KeyFrames;
    IBAGlobalParams iba_params;
    listdir(RawPointCloudFiles, pointcloud_dir);
    // IO Config
    const std::string TwcFile = base_dir + io_config["VOFile"].as<std::string>();
    const std::string TwlFile = base_dir + io_config["LOFile"].as<std::string>();
    const std::string resFile = base_dir + io_config["ResFile"].as<std::string>();
    const std::string KyeFrameIdFile = base_dir + io_config["VOIdFile"].as<std::string>();
    const std::string initSim3File = base_dir + io_config["init_sim3"].as<std::string>();
    const std::string gtSim3File = base_dir + io_config["gt_sim3"].as<std::string>();
    assert(file_exist(initSim3File));
    assert(file_exist(gtSim3File));
    // ORB Config
    const std::string ORBVocFile = orb_config["Vocabulary"].as<std::string>();
    const std::string ORBCfgFile = orb_config["Config"].as<std::string>();
    std::string ORBKeyFrameDir = orb_config["KeyFrameDir"].as<std::string>();
    checkpath(ORBKeyFrameDir);
    const std::string ORBMapFile = orb_config["MapFile"].as<std::string>();
    // runtime config
    iba_params.max_pixel_dist = runtime_config["max_pixel_dist"].as<double>();
    iba_params.num_best_covis = runtime_config["num_best_covis"].as<int>();
    iba_params.min_covis_weight = runtime_config["min_covis_weight"].as<int>();
    iba_params.kdtree2d_max_leaf_size = runtime_config["kdtree2d_max_leaf_size"].as<int>();
    iba_params.kdtree3d_max_leaf_size = runtime_config["kdtree3d_max_leaf_size"].as<int>();
    iba_params.corr_3d_2d_threshold = runtime_config["corr_3d_2d_threshold"].as<double>();
    iba_params.corr_3d_3d_threshold = runtime_config["corr_3d_3d_threshold"].as<double>();
    iba_params.norm_max_pts = runtime_config["norm_max_pts"].as<int>();
    iba_params.norm_min_pts = runtime_config["norm_min_pts"].as<int>();
    iba_params.norm_radius = runtime_config["norm_radius"].as<double>();
    iba_params.norm_reg_threshold = runtime_config["norm_reg_threshold"].as<double>();
    iba_params.min_diff_dist = runtime_config["min_diff_dist"].as<double>();
    iba_params.he_threshold = runtime_config["he_threshold"].as<double>();
    iba_params.err_weight = runtime_config["err_weight"].as<std::vector<double>>();
    iba_params.lb = runtime_config["lb"].as<std::vector<double>>();
    iba_params.ub = runtime_config["ub"].as<std::vector<double>>();
    iba_params.PointCloudskip = io_config["PointCloudskip"].as<int>();
    iba_params.PointCloudOnlyPositiveX = io_config["PointCloudOnlyPositiveX"].as<bool>();
    iba_params.max_bbeval = runtime_config["max_bbeval"].as<int>();
    iba_params.valid_rate = runtime_config["valid_rate"].as<double>();
    iba_params.verborse = runtime_config["verborse"].as<bool>();
    iba_params.use_plane = runtime_config["use_plane"].as<bool>();
    const bool use_cache = runtime_config["use_cache"].as<bool>();
    const bool use_vns = runtime_config["use_vns"].as<bool>();
    const std::string direction_type = runtime_config["direction_type"].as<std::string>();
    const int seed = runtime_config["seed"].as<int>();
    const std::string NomadCacheFile = runtime_config["cacheFile"].as<std::string>();
    const double min_mesh = runtime_config["min_mesh"].as<double>();
    const std::vector<double> init_frame_size = runtime_config["init_frame"].as<std::vector<double>>();
    std::vector<double> min_mesh_size(7, min_mesh);
    YAML::Node FrameIdCfg = YAML::LoadFile(KyeFrameIdFile);
    std::vector<int> vKFFrameId = FrameIdCfg["mnFrameId"].as<std::vector<int>>();
    
    std::vector<Eigen::Isometry3d> RawPointCloudPoses, PointCloudPoses;
    ReadPoseList(TwlFile, RawPointCloudPoses);
    PointCloudPoses.reserve(vKFFrameId.size());
    if(vKFFrameId[0]==0)
        for(auto &KFId: vKFFrameId)
            PointCloudPoses.push_back(RawPointCloudPoses[KFId]);
    else
    {
        Eigen::Isometry3d refPose = RawPointCloudPoses[vKFFrameId[0]].inverse();
        for(auto &KFId: vKFFrameId)
            PointCloudPoses.push_back(refPose * RawPointCloudPoses[KFId]);
    }
    std::vector<VecVector3d> PointClouds;
    PointClouds.resize(vKFFrameId.size());
    
    int numFiles = 0;
    std::printf("Loading %ld PointClouds\n",vKFFrameId.size());
    #pragma omp parallel for schedule(static)
    for(std::size_t i = 0; i < vKFFrameId.size(); ++i)
    {
        VecVector3d PointCloud;
        readPointCloud(pointcloud_dir + RawPointCloudFiles[vKFFrameId[i]], PointCloud);
        PointClouds[i] = std::move(PointCloud);
        #pragma omp critical
        {
            ++numFiles;
            if(iba_params.verborse && numFiles % 100 == 0)
                std::printf("Read %0.2lf %% PointClouds\n", 100.0*numFiles/vKFFrameId.size());
        }
    }
    ORB_SLAM2::System orb_slam(ORBVocFile, ORBCfgFile, ORB_SLAM2::System::MONOCULAR, false);
    orb_slam.Shutdown(); // Do not need any ORB running threads
    orb_slam.RestoreSystemFromFile(ORBKeyFrameDir, ORBMapFile);
    KeyFrames = orb_slam.GetAllKeyFrames(true);
    std::sort(KeyFrames.begin(), KeyFrames.end(), ORB_SLAM2::KeyFrame::lId);
    std::unordered_map<int, int> KFIdMap; // KeyFrame mnId -> FileIndex
    KFIdMap.reserve(KeyFrames.size());
    for(std::size_t i = 0; i < KeyFrames.size(); ++i)
        KFIdMap.insert(std::make_pair(KeyFrames[i]->mnId, i));
    Eigen::Matrix4d rigid;
    double scale;
    std::tie(rigid, scale) = readSim3(initSim3File);
    std::cout << "initial Result: " << std::endl;
    std::cout << "Rotation:\n" << rigid.topLeftCorner(3, 3) << std::endl;
    std::cout << "Translation: " << rigid.topRightCorner(3, 1).transpose() << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    g2o::SE3Quat quat(rigid.topLeftCorner(3, 3), rigid.topRightCorner(3, 1));
    g2o::Vector7 offset;
    offset.head<6>() = quat.log();
    offset(6) = scale;
    std::tie(rigid, scale) = readSim3(gtSim3File);
    std::cout << "GT Result: " << std::endl;
    std::cout << "Rotation:\n" << rigid.topLeftCorner(3, 3) << std::endl;
    std::cout << "Translation: " << rigid.topRightCorner(3, 1).transpose() << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    g2o::SE3Quat gt_quat(rigid.topLeftCorner(3, 3), rigid.topRightCorner(3, 1));
    g2o::Vector7 gtlog;
    gtlog.head<6>() = gt_quat.log();
    gtlog(6) = scale;
    std::vector<double> x0(offset.data(), offset.data() + offset.size());
    std::vector<double> gtx0(gtlog.data(), gtlog.data() + gtlog.size());
    g2o::Vector7 range_lb(iba_params.lb.data());
    g2o::Vector7 range_ub(iba_params.ub.data());
    range_lb = offset + range_lb;
    range_ub = offset + range_ub;
    std::cout << "NLopt x0:" << offset.transpose() << std::endl;
    std::cout << "gt:" << gtlog.transpose() << std::endl;
    std::cout << "NLopt lb: " << range_lb.transpose() << std::endl;
    std::cout << "NLopt ub: " << range_ub.transpose() << std::endl;
    std::vector<double> lb(range_lb.data(), range_lb.data() + range_lb.size());
    std::vector<double> ub(range_ub.data(), range_ub.data() + range_ub.size());
    std::vector<std::pair<std::string, std::vector<double>>> special_points;
    special_points.reserve(4);
    special_points.push_back(std::make_pair("x0",x0));
    special_points.push_back(std::make_pair("gt",gtx0));
    special_points.push_back(std::make_pair("lb",lb));
    special_points.push_back(std::make_pair("ub",ub));
    // Set NOMAD Parameters
    auto nomad_params = std::make_shared<NOMAD::AllParameters>();
    auto nomad_type = NOMAD::EvalType::BB;
    nomad_params->set_MAX_BB_EVAL(iba_params.max_bbeval);
    nomad_params->set_DIMENSION(7);
    NOMAD::BBOutputTypeList bbOutputTypes;
    bbOutputTypes.push_back(NOMAD::BBOutputType::OBJ);
    bbOutputTypes.push_back(NOMAD::BBOutputType::PB); // Relaxible Constraint |he_err| < delta
    bbOutputTypes.push_back(NOMAD::BBOutputType::PB); // Relaxible Constraint
    bbOutputTypes.push_back(NOMAD::BBOutputType::PB); // Corr Valid Rate > threshold
    nomad_params->set_BB_OUTPUT_TYPE(bbOutputTypes);
    NOMAD::ArrayOfDouble nomad_lb(lb), nomad_ub(ub);
    nomad_params->set_LOWER_BOUND(nomad_lb);
    nomad_params->set_UPPER_BOUND(nomad_ub);
    NOMAD::Point nomad_x0(x0); NOMAD::Point nomad_gtx0(gtx0);
    nomad_params->set_X0(nomad_x0);
    nomad_params->setAttributeValue("DISPLAY_STATS", NOMAD::ArrayOfString("BBE ( SOL ) BBO"));
    nomad_params->set_SEED(seed);
    if(use_cache)
        nomad_params->setAttributeValue("CACHE_FILE", NomadCacheFile);
    NOMAD::ArrayOfDouble nomad_mesh_minsize(min_mesh_size), nomad_frame_init(init_frame_size);
    nomad_params->set_INITIAL_POLL_SIZE(nomad_frame_init);
    nomad_params->set_MIN_MESH_SIZE(nomad_mesh_minsize);
    nomad_params->setAttributeValue("VNS_MADS_SEARCH",use_vns);
    if(direction_type=="N+1 QUAD")
        nomad_params->setAttributeValue("DIRECTION_TYPE",NOMAD::DirectionType::ORTHO_NP1_QUAD);
    else if(direction_type=="2N")
        nomad_params->setAttributeValue("DIRECTION_TYPE",NOMAD::DirectionType::ORTHO_2N);
    else if(direction_type=="N+1 NEG")
        nomad_params->setAttributeValue("DIRECTION_TYPE",NOMAD::DirectionType::ORTHO_NP1_NEG);
    else
        std::printf("\033[33;1mUnknown Direction Type:%s, set to the default:ORTHO_NP1_QUAD\033[0m",direction_type.c_str());
    nomad_params->checkAndComply();
    auto ev = std::make_unique<BALoss>(nomad_params->getEvalParams(), nomad_type,
        &PointCloudPoses, &PointClouds, &KFIdMap, &KeyFrames, &iba_params, special_points);
    NOMAD::MainStep nomad_main_step;
    nomad_main_step.setAllParameters(nomad_params);
    nomad_main_step.setEvaluator(std::move(ev));
    nomad_main_step.start();
    std::cout << "NOMAD Start\n";
    nomad_main_step.run();
    nomad_main_step.end();
    std::vector<NOMAD::EvalPoint> evalPointFeasList;
    auto nbFeas = NOMAD::CacheBase::getInstance()->findBestFeas(evalPointFeasList, NOMAD::Point(), 
        nomad_type, NOMAD::ComputeType::STANDARD, nullptr);
    NOMAD::EvalPoint evalPointFeas;
    if (nbFeas > 0)
    {
        evalPointFeas = evalPointFeasList[0];
    }
    g2o::Vector7 res_log;
    for(int i = 0; i < 7; ++i)
        res_log[i] = evalPointFeas[i].todouble();
    Eigen::Matrix3d optRotation;
    Eigen::Vector3d optTranslation;
    double optScale;
    std::tie(optRotation, optTranslation, optScale) = Sim3Exp<double>(res_log.data());
    rigid.setIdentity();
    rigid.topLeftCorner(3,3) = optRotation;
    rigid.topRightCorner(3,1) = optTranslation;
    std::cout << "Optimized Result: " << std::endl;
    std::cout << "Rotation:\n" << optRotation << std::endl;
    std::cout << "Translation: " << optTranslation.transpose() << std::endl;
    std::cout << "Scale: " << optScale << std::endl;
    writeSim3(resFile, rigid, optScale);
    std::printf("Result saved to %s\n",resFile.c_str());
    return 0;

}