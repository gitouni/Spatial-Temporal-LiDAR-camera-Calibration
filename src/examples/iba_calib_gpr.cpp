#include <limits>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include "g2o/core/robust_kernel_impl.h"
#include <yaml-cpp/yaml.h>
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "orb_slam/include/System.h"
#include "orb_slam/include/KeyFrame.h"
#include <opencv2/core/eigen.hpp>
#include "io_tools.h"
#include "kitti_tools.h"
#include "pointcloud.h"
#include "IBACalib2.hpp"
#include <mutex>
#include <unordered_set>
#include <omp.h>

typedef std::uint32_t IndexType; // other types cause error, why?
typedef std::vector<IndexType> VecIndex;
typedef std::pair<IndexType, IndexType> CorrType;
typedef std::vector<CorrType> CorrSet;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector2d, double, 2, nanoflann::metric_L2_Simple, IndexType> KDTree2D;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector3d, double, 3, nanoflann::metric_L2_Simple, IndexType> KDTree3D;


class IBAGPRParams{
public:
    IBAGPRParams(){};
public:
    double max_pixel_dist = 1.5;
    int min_covis_weight = 150;
    int kdtree2d_max_leaf_size = 10;
    int kdtree3d_max_leaf_size = 30;
    double neigh_radius = 0.6;
    int neigh_max_pts = 30;
    double robust_kernerl_delta = 2.98;
    double init_sigma = 10.;
    double init_l = 10.;
    double sigma_noise = 1e-10;
    bool optimize = true;
    bool verborse = true;
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
    ProjectPC.reserve(vKeyUnEigen.size());  // Actually, much larger than its size
    for (std::size_t i = 0; i < points.size(); ++i)
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
        kdtree->index->findNeighbors(resultSet, vKeyUnEigen[i].data(), nanoflann::SearchParameters(0.0F, true));
        if(resultSet.size() > 0 && sq_dist[0] <= max_corr_dist * max_corr_dist)
            corrset.push_back(std::make_pair(i, ProjectIndex[indices[0]]));
    }
}

/**
 * @brief Compute normal of each point in points indexed by indices
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param querys indices of selected points
 * @param radius radius to compute point covariance
 * @param max_pts max points of neighbours
 * @return std::tuple<VecVector3d, VecIndex>  valid normals, valid indices of querys
 */
std::tuple<VecVector3d, VecIndex> ComputeLocalNormal(const VecVector3d &points, const KDTree3D* kdtree, const VecIndex &querys,
    const double radius, const int max_pts)
{
    VecVector3d normals;
    VecIndex valid_indices;
    normals.reserve(querys.size());
    valid_indices.reserve(querys.size());
    for (std::size_t i = 0; i < querys.size(); ++i){
        const IndexType idx = querys[i];
        VecIndex indices(max_pts);
        std::vector<double> sq_dist(max_pts);
        nanoflann::KNNResultSet<double, IndexType> resultSet(max_pts);
        resultSet.init(indices.data(), sq_dist.data());
        kdtree->index->findNeighbors(resultSet, points[idx].data(), nanoflann::SearchParameters());
        std::size_t k = resultSet.size();
        if(k < 3)
            continue;
        k = std::distance(sq_dist.begin(),
                      std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                       radius * radius)); // iterator difference between start and last valid index
        indices.resize(k);
        sq_dist.resize(k);
        if(k < 3)
            continue;
        Eigen::Matrix3d covariance = ComputeCovariance(points, indices);
        Eigen::Vector3d normal, eval;
        std::tie(normal, eval) = FastEigen3x3_EV(covariance);
        std::vector<double> eigenvalues{eval[0], eval[1], eval[2]};
        std::sort(eigenvalues.begin(), eigenvalues.end());
        if(!(eigenvalues[2] > 3 * eigenvalues[1] && eigenvalues[2] > 3 * eigenvalues[0] && eigenvalues[2] > 1e-2))
            continue; // Valid Plane Verification
        normals.push_back(normal); // self-to-self distance is minimum
        valid_indices.push_back(i);
    }
    return {normals, valid_indices};
}


/**
 * @brief Compute neighbors of of each point in points indexed by indices
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param indices indices of selected points
 * @param radius radius to compute point covariance
 * @param max_pts max points of neighbours
 * @return std::tuple<std::vector<VecIndex>, VecIndex> Vec of neighbors' indices, valid index of querys
 */
std::tuple<std::vector<VecIndex>, VecIndex> ComputeLocalNeighbor(const VecVector3d &points, const KDTree3D* kdtree, const VecIndex &querys,
    const double radius, const int max_pts)
{
    std::vector<VecIndex> neighbors;
    neighbors.reserve(querys.size());
    VecIndex valid_indices;
    valid_indices.reserve(querys.size());
    for (std::size_t i = 0; i < querys.size(); ++i){
        const IndexType query_idx = querys[i];
        VecIndex indices(max_pts);
        std::vector<double> sq_dist(max_pts);
        nanoflann::KNNResultSet<double, IndexType> resultSet(max_pts);
        resultSet.init(indices.data(), sq_dist.data());
        kdtree->index->findNeighbors(resultSet, points[query_idx].data(), nanoflann::SearchParameters());
        int k = resultSet.size();
        if(k < 3)
            continue;
        k = std::distance(sq_dist.begin(),
                      std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                       radius * radius)); // iterator difference between start and last valid index
        indices.resize(k);
        sq_dist.resize(k);
        if(k < 3)
            continue;
        neighbors.push_back(indices); // self-to-self distance is minimum
        valid_indices.push_back(i);
    }
    return {neighbors, valid_indices};
}

void initInput(double* params, const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames, const g2o::Vector7& init_sim3_log, std::unordered_map<int, int> &KFIdMap)
{
    KFIdMap.reserve(KeyFrames.size());
    std::copy(init_sim3_log.data(), init_sim3_log.data()+7, params);
    int posei = 0;
    for(auto &pKF:KeyFrames)
    {
        cv::Mat pose = pKF->GetPose();
        Eigen::Matrix4d poseEigen;
        cv::cv2eigen(pose, poseEigen);
        g2o::Vector6 se3_log = SE3Log(poseEigen.topLeftCorner(3, 3), poseEigen.topRightCorner(3, 1));
        std::copy(se3_log.data(), se3_log.data() + 6, params + 7 + posei*6);
        KFIdMap.insert(std::make_pair(int(pKF->mnId), posei));
        posei += 1;
    }
}


void BuildProblem(const std::vector<VecVector3d> &PointClouds, const std::vector<std::unique_ptr<KDTree3D>> &PointKDTrees, const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames,
    const std::unordered_map<int, int> &KFIdMap, double* params, ceres::Problem &problem, const IBAGPRParams &iba_params)
{
    
    int cnt = 0;
    int edge_cnt = 0;
    double* const pose = params + 7;  // pose pointer position
    Eigen::Matrix3d init_rotation;
    Eigen::Vector3d init_translation;
    double init_scale;
    std::tie(init_rotation, init_translation, init_scale) = Sim3Exp<double>(params);
    Eigen::Matrix4d initSE3_4x4(Eigen::Matrix4d::Identity());
    initSE3_4x4.topLeftCorner(3, 3) = init_rotation;
    initSE3_4x4.topRightCorner(3, 1) = init_translation;
    Eigen::Isometry3d initSE3;
    initSE3.matrix() = initSE3_4x4;
    #pragma omp parallel for schedule(static) reduction(+:cnt)
    for(std::size_t Fi = 0; Fi < PointClouds.size(); ++Fi)
    {
        VecVector3d points = PointClouds[Fi];  // pointcloud in camera coord
        VecVector3d normals;  // point normals in camera coord
        ORB_SLAM2::KeyFrame* pKF = KeyFrames[Fi];
        const double H = pKF->mnMaxY, W = pKF->mnMaxX;
        TransformPointCloudInplace(points, initSE3); // Transfer to Camera coordinate
        CorrSet corrset; // pair of (idx of image points, idx of pointcloud points)
        FindProjectCorrespondences(points, pKF, iba_params.kdtree2d_max_leaf_size, iba_params.max_pixel_dist, corrset);
        if(corrset.size() < 50)
            continue;
        std::vector<ORB_SLAM2::KeyFrame*> ConvisKeyFrames = pKF->GetCovisiblesByWeightSafe(iba_params.min_covis_weight);  // for debug
        std::vector<std::map<int, int>> KptMapList; // Keypoint-Keypoint Corr
        std::vector<Eigen::Matrix4d> relPoseList; // RelPose From Reference to Convisible KeyFrames
        std::set<int> srcKptIndices;  // Matched Keypoints in the Reference KeyFrame
        KptMapList.reserve(ConvisKeyFrames.size());
        relPoseList.reserve(ConvisKeyFrames.size());
        const cv::Mat invRefPose = pKF->GetPoseInverseSafe();
        for(auto pKFConv:ConvisKeyFrames)
        {
            auto KptMap = pKF->GetMatchedKptIds(pKFConv);
            for(auto &kpt_pair:KptMap)
                srcKptIndices.insert(kpt_pair.first);
            cv::Mat relPose = pKFConv->GetPose() * invRefPose;  // Transfer from c1 coordinate to c2 coordinate
            Eigen::Matrix4d relPoseEigen;
            cv::cv2eigen(relPose, relPoseEigen);
            KptMapList.push_back(std::move(KptMap));
            relPoseList.push_back(std::move(relPoseEigen));
        }
        VecIndex indices; // valid indices for projecion
        indices.reserve(corrset.size());
        VecIndex subindices; // valid subindices of `indices` through normal computing
        std::vector<VecIndex> neighbor_indices;
        std::tie(neighbor_indices, subindices) = ComputeLocalNeighbor(PointClouds[Fi], PointKDTrees[Fi].get(), indices, iba_params.neigh_radius, iba_params.neigh_max_pts);
        for(std::size_t sub_idx = 0; sub_idx < subindices.size(); ++ sub_idx)
        {
            const IndexType idx = subindices[sub_idx];  // valid idx in corrset
            const VecIndex &neigh_idx = neighbor_indices[sub_idx];
            VecVector3d neighbor_pts;
            neighbor_pts.reserve(neigh_idx.size());
            for(auto const &idx:neigh_idx)
                neighbor_pts.push_back(points[idx]);
            const int point2d_idx = corrset[idx].first;  // KeyPoint Idx matched with PointCloud
            const int point3d_idx = corrset[idx].second; // Point Idx matched with KeyPoints
            double u0 = pKF->mvKeysUn[point2d_idx].pt.x;
            double v0 = pKF->mvKeysUn[point2d_idx].pt.y;
            // transform 3d point back to LiDAR coordinate
            Eigen::Vector3d p0 = initSE3.inverse() * points[point3d_idx];  // cooresponding point (LiDAR coord)
            Eigen::Vector3d n0 = initSE3_4x4.topLeftCorner(3, 3).transpose() * normals[sub_idx];  // cooresponding point normal (LiDAR coord)
            std::vector<double> u1_list, v1_list;
            std::vector<double*> param_blocks;
            param_blocks.push_back(params); // extrinsic log
            param_blocks.push_back(params + 7 + 6*Fi); // Reference Pose
            for(std::size_t pKFConvi = 0; pKFConvi < ConvisKeyFrames.size(); ++pKFConvi){
                auto pKFConv = ConvisKeyFrames[pKFConvi];
                // Skip if Cannot Find this 2d-3d matching map in Keypoint-to-Keypoint matching map
                if(KptMapList[pKFConvi].count(point2d_idx) == 0)
                    continue;
                const int convis_idx = KptMapList[pKFConvi][point2d_idx];  // corresponding KeyPoints idx in a convisible KeyFrame
                double u1 = pKFConv->mvKeysUn[convis_idx].pt.x;
                double v1 = pKFConv->mvKeysUn[convis_idx].pt.y;
                Eigen::Matrix4d relPose = relPoseList[pKFConvi];  // Twc2 * inv(Twc1)
                u1_list.push_back(u1);
                v1_list.push_back(v1);
                int ConvFi = KFIdMap.at(pKFConv->mnId);
                param_blocks.push_back(params + 7 + 6*ConvFi);
            }
            if(u1_list.size() == 0)
                continue; // Should Not Run into
            ceres::LossFunction *loss_function = new ceres::HuberLoss(iba_params.robust_kernerl_delta * u1_list.size());
            ceres::CostFunction *cost_function = UIBA_GPRFactor::Create(
                iba_params.init_sigma, iba_params.init_l, iba_params.sigma_noise, neighbor_pts, pKF->fx, pKF->fy,
                pKF->cx, pKF->cy, u0, v0, u1_list, v1_list, iba_params.optimize, iba_params.verborse
            );
            problem.AddResidualBlock(cost_function, loss_function, param_blocks);
            ++edge_cnt;  // valid factor count
        }
        ++cnt; // total iterations
        #pragma omp critical
        {
            if(iba_params.verborse && (cnt % 100 == 0)){
                char msg[80];
                std::sprintf(msg, "Problem Building: %0.2lf %% finished (%d iters | %d factors)", (100.0*cnt)/PointClouds.size(),cnt, edge_cnt);
                std::cout << msg << std::endl;
            }
        }
    }
    if(iba_params.verborse)
        std::cout << "" << edge_cnt << " factors has been added." << std::endl;
}


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
    std::vector<std::string> RawPointCloudFiles;
    listdir(RawPointCloudFiles, pointcloud_dir);
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
    IBAGPRParams iba_params;
    iba_params.max_pixel_dist = runtime_config["max_pixel_dist"].as<double>();
    iba_params.min_covis_weight = runtime_config["min_covis_weight"].as<int>();
    iba_params.kdtree2d_max_leaf_size = runtime_config["kdtree2d_max_leaf_size"].as<int>();
    iba_params.kdtree3d_max_leaf_size = runtime_config["kdtree3d_max_leaf_size"].as<int>();
    iba_params.neigh_radius = runtime_config["neigh_radius"].as<double>();
    iba_params.neigh_max_pts = runtime_config["neigh_max_pts"].as<int>();
    iba_params.robust_kernerl_delta = runtime_config["robust_kernerl_delta"].as<double>();
    const int max_iba_iter = runtime_config["max_iba_iter"].as<int>();
    const int inner_iba_iter = runtime_config["inner_iba_iter"].as<int>();
    iba_params.verborse = runtime_config["verborse"].as<bool>();

    YAML::Node FrameIdCfg = YAML::LoadFile(KyeFrameIdFile);
    std::vector<int> vKFFrameId = FrameIdCfg["mnFrameId"].as<std::vector<int>>();
    // Pre-allocate memory to enhance peformance
    std::vector<VecVector3d> PointClouds;
    std::vector<std::unique_ptr<KDTree3D>> PointKDTrees;
    PointClouds.resize(vKFFrameId.size());
    PointKDTrees.resize(vKFFrameId.size());
    std::size_t KFcnt = 0;
    #pragma omp parallel for schedule(static) reduction(+:KFcnt)
    for(std::size_t i = 0; i < vKFFrameId.size(); ++i)
    {
        auto KFId = vKFFrameId[i];
        VecVector3d PointCloud;
        readPointCloud(pointcloud_dir + RawPointCloudFiles[KFId], PointCloud);
        PointKDTrees[i] = std::make_unique<KDTree3D>(3, PointCloud, iba_params.kdtree3d_max_leaf_size);
        PointClouds[i] = std::move(PointCloud);
        KFcnt++;
        #pragma omp critical
        {
            if(iba_params.verborse && (KFcnt) % 100 ==0)
                std::printf("Read PointCloud %0.2lf %% with KDTree built.\n", 100.0*(KFcnt)/vKFFrameId.size());
        }
    }
    RawPointCloudFiles.clear(); 

    ORB_SLAM2::System orb_slam(ORBVocFile, ORBCfgFile, ORB_SLAM2::System::MONOCULAR, false);
    orb_slam.Shutdown(); // Do not need any ORB running threads
    orb_slam.RestoreSystemFromFile(ORBKeyFrameDir, ORBMapFile);
    std::vector<ORB_SLAM2::KeyFrame*> KeyFrames = orb_slam.GetAllKeyFrames(true);
    std::sort(KeyFrames.begin(), KeyFrames.end(), ORB_SLAM2::KeyFrame::lId);
    Eigen::Matrix4d rigid;
    double scale;
    std::tie(rigid, scale) = readSim3(initSim3File);
    std::cout << "initial Result: " << std::endl;
    std::cout << "Rotation:\n" << rigid.topLeftCorner(3, 3) << std::endl;
    std::cout << "Translation: " << rigid.topRightCorner(3, 1).transpose() << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    g2o::Vector7 init_sim3_log = Sim3Log(rigid.topLeftCorner(3, 3), rigid.topRightCorner(3, 1), scale);
    std::unordered_map<int, int> KFIdMap;
    double params[7 + 6 * KeyFrames.size()];
    
    initInput(params, KeyFrames, init_sim3_log, KFIdMap);
    Eigen::Matrix3d optimizedRotation; Eigen::Vector3d optimizedTranslation; double optimizedScale;
    for(int iba_iter = 0; iba_iter < max_iba_iter; ++ iba_iter)
    {
        ceres::Solver::Options options;
        options.max_num_iterations = 30;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = omp_get_max_threads(); // use all threads
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        ceres::Problem problem;
        BuildProblem(PointClouds, PointKDTrees, KeyFrames, KFIdMap, params, problem, iba_params);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if(iba_params.verborse)
        {
            std::tie(optimizedRotation, optimizedTranslation, optimizedScale) = Sim3Exp<double>(params);
            std::cout << "IBA iter = \033[33;1m" << iba_iter << "\033[0m" << std::endl;
            std::cout << "Rotation:\n" << optimizedRotation << std::endl;
            std::cout << "Translation: " << optimizedTranslation.transpose() << std::endl;
            std::cout << "Scale: " << optimizedScale << std::endl;
        }
    }
    std::tie(optimizedRotation, optimizedTranslation, optimizedScale) = Sim3Exp<double>(params);
    std::cout << "Final Result: " << std::endl;
    std::cout << "Rotation:\n" << optimizedRotation << std::endl;
    std::cout << "Translation: " << optimizedTranslation.transpose() << std::endl;
    std::cout << "Scale: " << optimizedScale << std::endl;
    rigid.topLeftCorner(3, 3) = optimizedRotation;
    rigid.topRightCorner(3, 1) = optimizedTranslation;
    writeSim3(resFile, rigid, optimizedScale);
    std::cout << "Sim3 Result saved to " << resFile << std::endl;
    
}