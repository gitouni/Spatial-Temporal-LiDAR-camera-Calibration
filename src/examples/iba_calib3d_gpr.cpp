#include <limits>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include "g2o/core/robust_kernel_impl.h"
#include <yaml-cpp/yaml.h>
#include "orb_slam/include/System.h"
#include "orb_slam/include/KeyFrame.h"
#include <opencv2/core/eigen.hpp>
#include "io_tools.h"
#include "kitti_tools.h"
#include "pointcloud.h"
#include "IBACalib2.hpp"
#include <mutex>
#include <unordered_set>
#include <thread>
#include <omp.h>

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
        kdtree->index->findNeighbors(resultSet, vKeyUnEigen[i].data(), nanoflann::SearchParameters());
        if(resultSet.size() > 0 && sq_dist[0] <= max_corr_dist * max_corr_dist)
            corrset.push_back(std::make_pair(i, ProjectIndex[indices[0]]));
    }
}



void BuildProblem(const std::vector<VecVector3d> &PointClouds, const std::vector<std::unique_ptr<KDTree3D>> &PointKDTrees,
    const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames, double* params, ceres::Problem &problem, const IBAGPR3dParams &iba_params)
{
    
    int iter_cnt = 0;
    int err_3d_2d_cnt = 0;
    int err_3d_3d_cnt = 0;
    double max_3d_dist2 = iba_params.max_3d_dist * iba_params.max_3d_dist;
    Eigen::Matrix3d init_rotation;
    Eigen::Vector3d init_translation;
    double init_scale;
    std::tie(init_rotation, init_translation, init_scale) = Sim3Exp<double>(params);
    Eigen::Matrix4d initSE3_4x4(Eigen::Matrix4d::Identity());
    initSE3_4x4.topLeftCorner(3, 3) = init_rotation;
    initSE3_4x4.topRightCorner(3, 1) = init_translation;
    Eigen::Isometry3d initSE3;
    initSE3.matrix() = initSE3_4x4;
    #pragma omp parallel for schedule(static)
    for(std::size_t Fi = 0; Fi < PointClouds.size(); ++Fi)
    {
        VecVector3d points;  // pointcloud in camera coord
        ORB_SLAM2::KeyFrame* pKF = KeyFrames[Fi];
        const double H = pKF->mnMaxY, W = pKF->mnMaxX;
        TransformPointCloud(PointClouds[Fi], points, initSE3);
        CorrSet corrset; // pair of (idx of image points, idx of pointcloud points)
        FindProjectCorrespondences(points, pKF, iba_params.kdtree2d_max_leaf_size, iba_params.max_pixel_dist, corrset);
        if(corrset.size() < iba_params.num_min_corr)
            continue;
        // build KeyPoint - MapPoint Mapping
        std::unordered_map<int, ORB_SLAM2::MapPoint*> mapKpt2Mpt;
        mapKpt2Mpt.reserve(pKF->mmapMpt2Kpt.size());
        for(auto const& [key, val] : pKF->mmapMpt2Kpt)
            mapKpt2Mpt[val] = key;
        std::vector<ORB_SLAM2::KeyFrame*> pCovisKFs;
        if(iba_params.num_best_covis > 0)
            pCovisKFs = pKF->GetBestCovisibilityKeyFramesSafe(iba_params.num_best_covis);  // for debug
        else
        {
            pCovisKFs = pKF->GetCovisiblesByWeightSafe(iba_params.min_covis_weight);
        }
        std::vector<std::map<int, int>> KptMapList; // Keypoint-Keypoint Corr
        std::vector<Eigen::Matrix4d> relPoseList; // RelPose From Reference to Covisible KeyFrames
        KptMapList.reserve(pCovisKFs.size());
        relPoseList.reserve(pCovisKFs.size());
        const cv::Mat invRefPose = pKF->GetPoseInverseSafe();
        for(auto pKFConv:pCovisKFs)
        {
            auto KptMap = pKF->GetMatchedKptIds(pKFConv);
            cv::Mat relPose = pKFConv->GetPose() * invRefPose;  // Transfer from c1 coordinate to c2 coordinate
            Eigen::Matrix4d relPoseEigen;
            cv::cv2eigen(relPose, relPoseEigen);
            KptMapList.push_back(std::move(KptMap));
            relPoseList.push_back(std::move(relPoseEigen));
        }
        VecIndex indices; // valid indices of PointCloud for projecion
        indices.reserve(corrset.size());
        VecIndex subindices; // valid subindices of `indices` through neighborhood search
        for(const CorrType &corr:corrset)
            indices.push_back(corr.second);
        std::vector<VecIndex> neighbor_indices;  // Vector of neighbors indices
        std::tie(neighbor_indices, subindices) = ComputeLocalNeighbor(PointClouds[Fi], PointKDTrees[Fi].get(), indices,
             iba_params.neigh_radius, iba_params.neigh_max_pts, iba_params.neigh_min_pts);
        for(std::size_t sub_idx = 0; sub_idx < subindices.size(); ++ sub_idx)
        {
            const IndexType valid_idx = subindices[sub_idx];  // valid idx in corrset
            const VecIndex &neigh_idx = neighbor_indices[sub_idx];
            VecVector3d neighbor_pts;
            neighbor_pts.reserve(neigh_idx.size());
            for(auto const &idx:neigh_idx)
                neighbor_pts.push_back(PointClouds[Fi][idx]);  // Lidar coord System
            
            const int point2d_idx = corrset[valid_idx].first;  // KeyPoint Idx matched with PointCloud
            if(mapKpt2Mpt.count(point2d_idx) == 0)
                continue;
            const int point3d_idx = corrset[valid_idx].second; // Point Idx matched with KeyPoints
            double u0 = pKF->mvKeysUn[point2d_idx].pt.x;
            double v0 = pKF->mvKeysUn[point2d_idx].pt.y;
            // select corresponding MapPoint of ORB_SLAM
            Eigen::Vector3d MapPoint;
            Eigen::Matrix4d Tcw;
            cv::cv2eigen(mapKpt2Mpt[point2d_idx]->GetWorldPos(), MapPoint);  // scaleless Pw in Ref Camera Coord (Tcw * Pw)
            cv::cv2eigen(pKF->GetPose(), Tcw);
            MapPoint = Tcw.topLeftCorner(3, 3) * MapPoint + Tcw.topRightCorner(3, 1);
            // transform 3d point back to LiDAR coordinate
            std::vector<double> u1_list, v1_list;
            std::vector<Eigen::Matrix3d> R_list;
            std::vector<Eigen::Vector3d> t_list;
            for(std::size_t pKFConvi = 0; pKFConvi < pCovisKFs.size(); ++pKFConvi){
                auto pKFConv = pCovisKFs[pKFConvi];
                // Skip if Cannot Find this 2d-3d matching map in Keypoint-to-Keypoint matching map
                if(KptMapList[pKFConvi].count(point2d_idx) == 0)
                    continue;
                const int covis_idx = KptMapList[pKFConvi][point2d_idx];  // corresponding KeyPoints idx in a covisible KeyFrame
                double u1 = pKFConv->mvKeysUn[covis_idx].pt.x;
                double v1 = pKFConv->mvKeysUn[covis_idx].pt.y;
                Eigen::Matrix4d relPose = relPoseList[pKFConvi];  // Twc2 * inv(Twc1)
                u1_list.push_back(u1);
                v1_list.push_back(v1);
                R_list.push_back(relPose.topLeftCorner(3, 3));
                t_list.push_back(relPose.topRightCorner(3, 1));  
            }
            if(u1_list.size() == 0)
                continue; // Should Not Run into
            #pragma omp critical
            {
                ceres::LossFunction *loss_3d_2d_func = new ceres::HuberLoss(iba_params.robust_kernel_delta);
                ceres::CostFunction *cost_3d_2d_func = IBA_GPRFactor::Create(
                    iba_params.init_sigma, iba_params.init_l, iba_params.sigma_noise, neighbor_pts, initSE3_4x4, pKF->mnMaxY, pKF->mnMaxX, pKF->fx, pKF->fy,
                    pKF->cx, pKF->cy, u0, v0, u1_list, v1_list, R_list, t_list, iba_params.optimize_gpr, false
                );
                problem.AddResidualBlock(cost_3d_2d_func, loss_3d_2d_func, params);
                ++err_3d_2d_cnt;  // valid factor count
            }
            Eigen::Vector3d MapPointInLidar = initSE3.inverse() * (MapPoint * init_scale);
            VecIndex MapPointIndices(1);
            std::vector<double> MapPointSqDist(1);
            nanoflann::KNNResultSet<double, IndexType> MapPointRes(1);
            MapPointRes.init(MapPointIndices.data(), MapPointSqDist.data());
            PointKDTrees[Fi]->index->findNeighbors(MapPointRes, MapPointInLidar.data(), nanoflann::SearchParameters());
            if(MapPointSqDist[0] > max_3d_dist2)
                continue;
            ceres::LossFunction *loss_3dfunction = new ceres::HuberLoss(iba_params.robust_kernel_3ddelta);
            IndexType MapPointQuery = MapPointIndices[0];
            Eigen::Vector3d MapPointNormal;
            Eigen::Vector3d MapPointNNPoint = PointClouds[Fi][MapPointQuery];
            bool state;
            std::tie(MapPointNormal, state) = ComputeLocalNormalSingle(PointClouds[Fi], PointKDTrees[Fi].get(), MapPointQuery, iba_params.norm_radius,
                iba_params.norm_max_pts, iba_params.norm_min_pts, iba_params.pvalue, iba_params.min_eval);
            
            #pragma omp critical
            {
                if(state)  // Point to Point Distance
                {
                    ceres::CostFunction *cost_3dfunction = Point2Plane_Factor::Create(MapPoint, MapPointNNPoint, MapPointNormal);
                    problem.AddResidualBlock(cost_3dfunction, loss_3dfunction, params);
                }
                else  // Point to Plane Distance
                {
                    ceres::CostFunction *cost_3dfunction = Point2Point_Factor::Create(MapPoint, MapPointNNPoint);
                    problem.AddResidualBlock(cost_3dfunction, loss_3dfunction, params);
                }
                ++err_3d_3d_cnt;  // valid factor count
            }

        }
        #pragma omp critical
        {
            ++iter_cnt; // total iterations
            if(iba_params.verborse && (iter_cnt % 100 == 0))
                std::printf("Problem Building: %0.2lf %% finished (%d iters | %d 3d-2d facotrs, %d 3d-3d factors)\n", (100.0*iter_cnt)/PointClouds.size(),iter_cnt, err_3d_2d_cnt, err_3d_3d_cnt);
        }
    }
    if(iba_params.verborse)
        std::printf("%d 3d-2d factos and %d 3d-3d factors have been added\n",err_3d_2d_cnt, err_3d_3d_cnt);
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
    file_exist(initSim3File);
    // ORB Config
    const std::string ORBVocFile = orb_config["Vocabulary"].as<std::string>();
    const std::string ORBCfgFile = orb_config["Config"].as<std::string>();
    std::string ORBKeyFrameDir = orb_config["KeyFrameDir"].as<std::string>();
    checkpath(ORBKeyFrameDir);
    const std::string ORBMapFile = orb_config["MapFile"].as<std::string>();
    // runtime config
    IBAGPR3dParams iba_params;
    iba_params.max_pixel_dist = runtime_config["max_pixel_dist"].as<double>();
    iba_params.num_best_covis = runtime_config["num_best_covis"].as<int>();
    iba_params.min_covis_weight = runtime_config["min_covis_weight"].as<int>();
    iba_params.kdtree2d_max_leaf_size = runtime_config["kdtree2d_max_leaf_size"].as<int>();
    iba_params.kdtree3d_max_leaf_size = runtime_config["kdtree3d_max_leaf_size"].as<int>();
    iba_params.neigh_radius = runtime_config["neigh_radius"].as<double>();
    iba_params.neigh_max_pts = runtime_config["neigh_max_pts"].as<int>();
    iba_params.init_sigma = runtime_config["init_sigma"].as<double>();
    iba_params.init_l = runtime_config["init_l"].as<double>();
    iba_params.sigma_noise = runtime_config["sigma_noise"].as<double>();
    iba_params.optimize_gpr = runtime_config["optimize_gpr"].as<bool>();
    iba_params.robust_kernel_delta = runtime_config["robust_kernel_delta"].as<double>();
    iba_params.PointCloudSkip = io_config["PointCloudSkip"].as<int>();
    iba_params.PointCloudOnlyPositiveX = io_config["PointCloudOnlyPositiveX"].as<bool>();
    const int max_iba_iter = runtime_config["max_iba_iter"].as<int>();
    const int inner_iba_iter = runtime_config["inner_iba_iter"].as<int>();
    iba_params.verborse = runtime_config["verborse"].as<bool>();

    YAML::Node FrameIdCfg = YAML::LoadFile(KyeFrameIdFile);
    std::vector<int> vKFFrameId = FrameIdCfg["mnFrameId"].as<std::vector<int>>();
    // Pre-allocate memory to enhance peformance
    std::printf("Read %ld PointClouds.\n",vKFFrameId.size());
    std::vector<VecVector3d> PointClouds;
    std::vector<std::unique_ptr<KDTree3D>> PointKDTrees;
    PointClouds.resize(vKFFrameId.size());
    PointKDTrees.resize(vKFFrameId.size());
    std::size_t KFcnt = 0;
    std::printf("Read %ld PointClouds.\n", PointClouds.size());
    #pragma omp parallel for schedule(static)
    for(std::size_t i = 0; i < vKFFrameId.size(); ++i)
    {
        auto KFId = vKFFrameId[i];
        VecVector3d PointCloud;
        readPointCloud(pointcloud_dir + RawPointCloudFiles[KFId], PointCloud,iba_params.PointCloudSkip, iba_params.PointCloudOnlyPositiveX);
        PointClouds[i] = std::move(PointCloud);
        #pragma omp critical
        {
            KFcnt++;
            if(iba_params.verborse && (KFcnt % 100 ==0))
                std::printf("Read PointCloud %0.2lf %%.\n", 100.0*(KFcnt)/vKFFrameId.size());
        }
    }
    RawPointCloudFiles.clear(); 
    std::printf("Build %ld KDTrees for PointCLouds.\n",vKFFrameId.size());
    KFcnt = 0;  // count again
    #pragma omp parallel for schedule(static)
    for(std::size_t i = 0; i < PointClouds.size(); ++i)
    {
        PointKDTrees[i] = std::make_unique<KDTree3D>(3, PointClouds[i], iba_params.kdtree3d_max_leaf_size);
        #pragma omp critical
        {
            KFcnt++;
            if(iba_params.verborse && (KFcnt % 100 ==0))
                std::printf("Build %0.2lf %% KDTrees.\n", 100.0*(KFcnt)/vKFFrameId.size());
        }
    }
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
    double params[7];
    std::copy(init_sim3_log.data(), init_sim3_log.data()+7, params);
    Eigen::Matrix3d optimizedRotation; Eigen::Vector3d optimizedTranslation; double optimizedScale;
    for(int iba_iter = 0; iba_iter < max_iba_iter; ++ iba_iter)
    {
        ceres::Solver::Options options;
        options.max_num_iterations = 30;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = std::thread::hardware_concurrency(); // use all threads
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        ceres::Problem problem;
        BuildProblem(PointClouds, PointKDTrees, KeyFrames, params, problem, iba_params);
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