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

/**
 * @brief Use the Reprojected Pixels of MapPoints as KeyPoint Position,
 *  find best 2d correspondence between reprojected pointcloud and keypoints point-by-point
 * 
 * @param points PointCloud (Camera Coord)
 * @param pKF Current KeyFrame pointer
 * @param leaf_size maximum leaf size of the 2d-kdtree
 * @param max_corr_dist maximum distance threshold to accept a 2d-2d correspondence
 * @return Map of Matched KeyPoints mnId <-> LidarPoints id
 */
CorrSet FindProjectCorrespondences(const VecVector3d &points, const ORB_SLAM2::KeyFrame* pKF,
    const Eigen::Matrix4d &Tcw, const std::unordered_map<int, std::vector<double>> &mappoint_map,
    const int leaf_size, const double max_corr_dist)
{
    CorrSet pt2d3d_map;
    std::unordered_map<int, Eigen::Vector2d> vKeyUnEigen;
    vKeyUnEigen.reserve(pKF->mmapMpt2Kpt.size());
    for(auto const &[pMapPoint,keypt_idx]:pKF->mmapMpt2Kpt)
    {
        Eigen::Vector3d MapPointPose(mappoint_map.at(static_cast<int>(pMapPoint->mnId)).data());
        MapPointPose = Tcw.topLeftCorner(3, 3) * MapPointPose + Tcw.topRightCorner(3, 1);
        Eigen::Vector2d Keypt;
        Keypt.x() = pKF->fx * MapPointPose.x() / MapPointPose.z() + pKF->cx;
        Keypt.y() = pKF->fy * MapPointPose.y() / MapPointPose.z() + pKF->cy;
        vKeyUnEigen[keypt_idx] = std::move(Keypt);
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
            if (0 <= u && u < W && 0 <= v && v < H){
                ProjectPC.push_back(Eigen::Vector2d(u, v));
                ProjectIndex.push_back(i);
            }
                
        }
    }
    std::unique_ptr<KDTree2D> kdtree (new KDTree2D(2, ProjectPC, leaf_size));
    for(auto const &[keypt_idx, KeyPoint]:vKeyUnEigen)
    {
        std::vector<IndexType> indices(1);
        std::vector<double> sq_dist(1, std::numeric_limits<double>::max());
        nanoflann::KNNResultSet<double, IndexType> resultSet(1);
        resultSet.init(indices.data(), sq_dist.data());
        kdtree->index->findNeighbors(resultSet, KeyPoint.data(), nanoflann::SearchParameters());
        if(resultSet.size() > 0 && sq_dist[0] <= max_corr_dist * max_corr_dist)
            pt2d3d_map.push_back(std::make_pair(static_cast<IndexType>(keypt_idx),indices[0]));
    }
    return pt2d3d_map;
}

/**
 * @brief initialize camera poses (N x 6), MapPoint poses (N, 3)
 * 
 * @param KeyFrames Vector of KeyFrame pointers
 * @param MapPoints Vector of Mappoint pointers
 * @return std::tuple<std::vector<std::vector<double>>,  std::vector<std::vector<double>>, std::unordered_map<int, int>>  
 * Vector of KeyFrames poses (Tcw), Vector of MapPoint poses (Tw)
 */

/**
 * @brief initialize camera poses (N x 6), MapPoint poses (N, 3)
 * 
 * @param KeyFrames Vector of KeyFrame pointers
 * @param MapPoints Vector of Mappoint pointers
 * @return std::tuple<std::vector<std::vector<double>>,  std::unordered_map<int, std::vector<double>>> Vector of KeyFrames poses (Tcw), Map of MapPoint poses (mnId -> Tw)
 */
std::tuple<std::vector<std::vector<double>>,  std::unordered_map<int, std::vector<double>>> initInput(
    const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames, const std::vector<ORB_SLAM2::MapPoint*> &MapPoints)
{
    std::vector<std::vector<double>> pose_list;
    std::unordered_map<int, std::vector<double>> map_mptpose;
    pose_list.reserve(KeyFrames.size());
    map_mptpose.reserve(MapPoints.size());
    for(int pKFi = 0; pKFi < static_cast<int>(KeyFrames.size()); ++pKFi)
    {
        auto const pKF = KeyFrames[pKFi];
        cv::Mat pose = pKF->GetPose();
        Eigen::Matrix4d poseEigen;
        cv::cv2eigen(pose, poseEigen);
        g2o::Vector6 se3_log = SE3Log(poseEigen.topLeftCorner(3, 3), poseEigen.topRightCorner(3, 1));
        std::vector<double> camera_pose(se3_log.data(), se3_log.data() + 6);
        pose_list.push_back(std::move(camera_pose));
    }
    for(int pMpti = 0; pMpti < static_cast<int>(MapPoints.size()); ++pMpti)
    {
        auto const pMpt = MapPoints[pMpti];
        Eigen::Vector3d MapPointPose;
        cv::cv2eigen(pMpt->GetWorldPosSafe(), MapPointPose);
        std::vector<double> MptPoseVec(MapPointPose.data(), MapPointPose.data() + 3);
        map_mptpose[static_cast<int>(pMpt->mnId)] = std::move(MptPoseVec);
    }
    return {pose_list, map_mptpose};
}

void BuildProblem(const std::vector<VecVector3d> &PointClouds, const std::vector<std::unique_ptr<KDTree3D>> &PointKDTrees,
 const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames, std::vector<double> &init_sim3_log, std::vector<std::vector<double>> &pose_params,
 std::unordered_map<int, std::vector<double>> &mappoint_params, ceres::Problem &problem, const IBAGPR3dParams &iba_params)
{
    int iter_cnt = 0;
    int err_3d_2d_cnt = 0;
    int err_3d_3d_cnt = 0;
    double max_3d_dist2 = iba_params.max_3d_dist * iba_params.max_3d_dist;
    Eigen::Matrix3d init_rotation;
    Eigen::Vector3d init_translation;
    double init_scale;
    std::tie(init_rotation, init_translation, init_scale) = Sim3Exp<double>(init_sim3_log.data());
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
        // build KeyPoint - MapPoint Mapping
        std::unordered_map<int, ORB_SLAM2::MapPoint*> mapKpt2Mpt;
        mapKpt2Mpt.reserve(pKF->mmapMpt2Kpt.size());
        for(auto const &[pMpt, keypt_idx] : pKF->mmapMpt2Kpt)
            mapKpt2Mpt[keypt_idx] = pMpt;
        TransformPointCloud(PointClouds[Fi], points, initSE3);
        Eigen::Matrix3d Rcw;
        Eigen::Vector3d tcw;
        Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
        std::tie(Rcw, tcw) = SE3Exp<double>(pose_params[Fi].data());
        Tcw.topLeftCorner(3,3) = Rcw;
        Tcw.topRightCorner(3,1) = tcw;
        CorrSet pt2d3d_indices = FindProjectCorrespondences(points, pKF, Tcw, mappoint_params, iba_params.kdtree2d_max_leaf_size, iba_params.max_pixel_dist);
        if(pt2d3d_indices.size() < iba_params.num_min_corr)
            continue;
        for(auto const &[point2d_idx, point3d_idx]:pt2d3d_indices)
        {
            double u0 = pKF->mvKeysUn[point2d_idx].pt.x;
            double v0 = pKF->mvKeysUn[point2d_idx].pt.y;
            // Should not happen
            // if(mapKpt2Mpt.count(point2d_idx) == 0)
            //     continue;
            auto const &pMpt = mapKpt2Mpt.at(point2d_idx);
            /* Compute Bundle Adjustment Error */
            ceres::LossFunction *loss_ba_func = new ceres::HuberLoss(iba_params.robust_kernel_delta);
            ceres::CostFunction *cost_ba_func = BA_Factor::Create(u0, v0, pKF->fx, pKF->fy, pKF->cx, pKF->cy);
            #pragma omp critical
            {
                problem.AddResidualBlock(cost_ba_func, loss_ba_func, pose_params[Fi].data(), mappoint_params[static_cast<int>(pMpt->mnId)].data());
                ++err_3d_2d_cnt;
            }
            /* Compute Lidar-Camera 3d-3d error */
            // select corresponding MapPoint of ORB_SLAM
            Eigen::Vector3d MapPoint(mappoint_params[static_cast<int>(pMpt->mnId)].data());

            MapPoint = Rcw * MapPoint + tcw;
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
            Eigen::Vector3d QueryPoint = PointClouds[Fi][MapPointQuery];
            bool valid_plane;
            std::tie(MapPointNormal, valid_plane) = ComputeLocalNormalSingle(PointClouds[Fi], PointKDTrees[Fi].get(), MapPointQuery, iba_params.norm_radius,
                iba_params.norm_max_pts, iba_params.norm_min_pts, iba_params.pvalue, iba_params.min_eval);
            // TODO: involve pose_params and mappoint_params in optimization
            #pragma omp critical
            {
                if(valid_plane)  // Point to Point Distance
                {
                    ceres::CostFunction *cost_3dfunction = CrossPL_Factor::Create(QueryPoint, MapPointNormal);
                    problem.AddResidualBlock(cost_3dfunction, loss_3dfunction, init_sim3_log.data(), pose_params[Fi].data(), mappoint_params[static_cast<int>(pMpt->mnId)].data());
                }
                else  // Point to Plane Distance
                {
                    ceres::CostFunction *cost_3dfunction = CrossPt_Factor::Create(QueryPoint);
                    problem.AddResidualBlock(cost_3dfunction, loss_3dfunction, init_sim3_log.data(), pose_params[Fi].data(), mappoint_params[static_cast<int>(pMpt->mnId)].data());
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
    iba_params.norm_max_pts = runtime_config["norm_max_pts"].as<int>();
    iba_params.norm_min_pts = runtime_config["norm_min_pts"].as<int>();
    iba_params.norm_radius = runtime_config["norm_radius"].as<double>();
    iba_params.min_diff_dist = runtime_config["min_diff_dist"].as<double>();
    iba_params.norm_reg_threshold = runtime_config["norm_reg_threshold"].as<double>();
    iba_params.kdtree2d_max_leaf_size = runtime_config["kdtree2d_max_leaf_size"].as<int>();
    iba_params.kdtree3d_max_leaf_size = runtime_config["kdtree3d_max_leaf_size"].as<int>();
    iba_params.robust_kernel_delta = runtime_config["robust_kernel_delta"].as<double>();
    iba_params.robust_kernel_3ddelta = runtime_config["robust_kernel_3ddelta"].as<double>();
    iba_params.PointCloudSkip = io_config["PointCloudSkip"].as<int>();
    iba_params.PointCloudOnlyPositiveX = io_config["PointCloudOnlyPositiveX"].as<bool>();
    iba_params.verborse = runtime_config["verborse"].as<bool>();
    const int max_iba_iter = runtime_config["max_iba_iter"].as<int>();
    const int inner_iba_iter = runtime_config["inner_iba_iter"].as<int>();
    const double iba_min_diff = runtime_config["iba_min_diff"].as<double>();
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
    auto KeyFrames = orb_slam.GetAllKeyFrames(true);
    auto MapPoints = orb_slam.GetAllMapPoints(true);
    std::sort(KeyFrames.begin(), KeyFrames.end(), ORB_SLAM2::KeyFrame::lId);
    std::sort(MapPoints.begin(), MapPoints.end(), ORB_SLAM2::MapPoint::lId);
    Eigen::Matrix4d rigid;
    double scale;
    std::tie(rigid, scale) = readSim3(initSim3File);
    std::cout << "initial Result: " << std::endl;
    std::cout << "Rotation:\n" << rigid.topLeftCorner(3, 3) << std::endl;
    std::cout << "Translation: " << rigid.topRightCorner(3, 1).transpose() << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    g2o::Vector7 g2o_sim3_log = Sim3Log(rigid.topLeftCorner(3, 3), rigid.topRightCorner(3, 1), scale);
    std::vector<double> init_sim3_log(g2o_sim3_log.data(), g2o_sim3_log.data() + 7);
    std::unordered_map<int, int> KFIdMap;  // Keyframe mnId to KeyFrame FileId
    std::unordered_map<int, int> MapptIdMap;  // MapPoint mnId to MapPoint FileId
    std::vector<std::vector<double>> pose_list;
    std::unordered_map<int, std::vector<double>> mappoint_map;
    std::tie(pose_list, mappoint_map) = initInput(KeyFrames, MapPoints);
    std::printf("Problem initilized with %ld KeyFrames and %ld MapPoints\n",KFIdMap.size(), MapptIdMap.size());
    Eigen::Matrix3d optimizedRotation; Eigen::Vector3d optimizedTranslation; double optimizedScale;
    std::vector<double> last_sim3_log(init_sim3_log);
    for(int iba_iter = 0; iba_iter < max_iba_iter; ++ iba_iter)
    {
        ceres::Solver::Options options;
        options.max_num_iterations = inner_iba_iter;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = std::thread::hardware_concurrency(); // use all threads
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        ceres::Problem problem;
        BuildProblem(PointClouds, PointKDTrees, KeyFrames, init_sim3_log, pose_list, mappoint_map, problem, iba_params);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if(iba_params.verborse)
        {
            std::tie(optimizedRotation, optimizedTranslation, optimizedScale) = Sim3Exp<double>(init_sim3_log.data());
            std::cout << "IBA iter = \033[33;1m" << iba_iter << "\033[0m" << std::endl;
            std::cout << "Rotation:\n" << optimizedRotation << std::endl;
            std::cout << "Translation: " << optimizedTranslation.transpose() << std::endl;
            std::cout << "Scale: " << optimizedScale << std::endl;
        }
        if(allClose(last_sim3_log, init_sim3_log, iba_min_diff))
        {
            std::cout << "Achieved Minimum Step: " << iba_min_diff << std::endl;
            break;
        }else
            last_sim3_log = init_sim3_log;
        
    }
    std::tie(optimizedRotation, optimizedTranslation, optimizedScale) = Sim3Exp<double>(init_sim3_log.data());
    std::cout << "Final Result: " << std::endl;
    std::cout << "Rotation:\n" << optimizedRotation << std::endl;
    std::cout << "Translation: " << optimizedTranslation.transpose() << std::endl;
    std::cout << "Scale: " << optimizedScale << std::endl;
    rigid.topLeftCorner(3, 3) = optimizedRotation;
    rigid.topRightCorner(3, 1) = optimizedTranslation;
    writeSim3(resFile, rigid, optimizedScale);
    std::cout << "Sim3 Result saved to " << resFile << std::endl;
    
}
