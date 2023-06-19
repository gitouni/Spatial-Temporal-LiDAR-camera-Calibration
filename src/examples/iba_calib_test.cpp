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
#include "IBACalib.hpp"
#include <mutex>
#include <set>

typedef std::uint32_t IndexType; // other types cause error, why?
typedef std::vector<IndexType> VecIndex;
typedef std::pair<IndexType, IndexType> CorrType;
typedef std::vector<CorrType> CorrSet;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector2d, double, 2, nanoflann::metric_L2_Simple, IndexType> KDTree2D;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector3d, double, 3, nanoflann::metric_L2_Simple, IndexType> KDTree3D;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<7, 1>> BlockSolverType; // the second template parameter is not used unless Schur Complement is used
typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType; // Linear Solver Type

class IBAPlaneParams{
public:
    IBAPlaneParams(){};
public:
    double max_pixel_dist = 1.5;
    int min_covis_weight = 100;
    int kdtree2d_max_leaf_size = 10;
    int kdtree3d_max_leaf_size = 30;
    double norm_radius = 0.6;
    int norm_max_pts = 30;
    double robust_kernel_delta = 2.98;
    double sq_err_threshold = 225.;
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


void BuildOptimizer(const std::vector<std::string> &PointCloudFiles, std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames,
    const g2o::Vector7 &init_sim3_log , g2o::SparseOptimizer &optimizer, const IBAPlaneParams &params)
{
    int cnt = 0;
    int edge_cnt = 0;
    optimizer.clear();
    VertexSim3* v = new VertexSim3();
    v->setEstimate(init_sim3_log);
    v->setId(0);
    optimizer.addVertex(v);
    Eigen::Matrix3d init_rotation;
    Eigen::Vector3d init_translation;
    double init_scale;
    std::tie(init_rotation, init_translation, init_scale) = Sim3Exp<double>(init_sim3_log.data());
    Eigen::Matrix4d initSE3_4x4(Eigen::Matrix4d::Identity());
    initSE3_4x4.topLeftCorner(3, 3) = init_rotation;
    initSE3_4x4.topRightCorner(3, 1) = init_translation;
    Eigen::Isometry3d initSE3;
    initSE3.matrix() = initSE3_4x4;
    std::mutex EdgeMutex;
    #pragma parallel for schedule(static)
    for(auto pair_it = std::make_pair(PointCloudFiles.begin(), KeyFrames.begin());
        pair_it.first != PointCloudFiles.end() && pair_it.second != KeyFrames.end(); pair_it.first++, pair_it.second++)
    {
        VecVector3d points, normals;
        readPointCloud(*(pair_it.first), points);
        ORB_SLAM2::KeyFrame* pKF = *(pair_it.second);
        const double H = pKF->mnMaxY, W = pKF->mnMaxX;
        TransformPointCloudInplace(points, initSE3); // Transfer to Camera coordinate
        CorrSet corrset; // pair of (idx of image points, idx of pointcloud points)
        FindProjectCorrespondences(points, pKF, params.kdtree2d_max_leaf_size, params.max_pixel_dist, corrset);
        if(corrset.size() < 50)
            continue;
        std::vector<ORB_SLAM2::KeyFrame*> ConvisKeyFrames = pKF->GetBestCovisibilityKeyFramesSafe(10);  // for debug
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

        for(const CorrType &corr:corrset)
        {
            if(srcKptIndices.count(corr.first) == 0)
                continue;
            const int point2d_idx = corr.first;  // KeyPoint Idx matched with PointCloud
            const int point3d_idx = corr.second; // Point Idx matched with KeyPoints
            double u0 = pKF->mvKeysUn[point2d_idx].pt.x;
            double v0 = pKF->mvKeysUn[point2d_idx].pt.y;
            // transform 3d point back to LiDAR coordinate
            Eigen::Vector3d p0 = initSE3.inverse() * points[point3d_idx];  // cooresponding point (LiDAR coord)
            for(std::size_t pKFConvi = 0; pKFConvi < ConvisKeyFrames.size(); ++pKFConvi){
                auto pKFConv = ConvisKeyFrames[pKFConvi];
                // Skip if Cannot Find this 2d-3d matching map in Keypoint-to-Keypoint matching map
                if(KptMapList[pKFConvi].count(point2d_idx) == 0)
                    continue;
                const int convis_idx = KptMapList[pKFConvi][point2d_idx];  // corresponding KeyPoints idx in a convisible KeyFrame
                double u1 = pKFConv->mvKeysUn[convis_idx].pt.x;
                double v1 = pKFConv->mvKeysUn[convis_idx].pt.y;
                Eigen::Matrix4d relPose = relPoseList[pKFConvi];  // Twc2 * inv(Twc1)
                IBATestEdge* e = new IBATestEdge(pKF->fx, pKF->fy, pKF->cx, pKF->cy,
                    u0, v0, u1, v1, p0, relPose.topLeftCorner(3, 3), relPose.topRightCorner(3, 1));
                e->setId(edge_cnt++);
                e->setVertex(0, v);
                e->setInformation(Eigen::Matrix2d::Identity());
                e->computeError();
                std::cout << e->chi2() << std::endl;
                g2o::RobustKernelHuber* rk(new g2o::RobustKernelHuber);
                rk->setDelta(params.robust_kernel_delta);
                e->setRobustKernel(rk);
                optimizer.addEdge(e);
            }
        }
        ++cnt;
        if(params.verborse && (cnt % 100 == 0)){
            std::unique_lock<std::mutex> lock(EdgeMutex);
            char msg[100];
            sprintf(msg, "Optimizer Building: %0.2lf %% finished (%d item %d edges)", (100.0*cnt)/PointCloudFiles.size(),cnt, edge_cnt);
            std::cout << msg << std::endl;
        }
    }
    if(params.verborse)
        std::cout << "" << edge_cnt << " edges has been added." << std::endl;
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
    std::vector<std::string> RawPointCloudFiles, PointCloudFiles;
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
    IBAPlaneParams params;
    params.max_pixel_dist = runtime_config["max_pixel_dist"].as<double>();
    params.min_covis_weight = runtime_config["min_covis_weight"].as<int>();
    params.kdtree2d_max_leaf_size = runtime_config["kdtree2d_max_leaf_size"].as<int>();
    params.kdtree3d_max_leaf_size = runtime_config["kdtree3d_max_leaf_size"].as<int>();
    params.norm_radius = runtime_config["norm_radius"].as<double>();
    params.norm_max_pts = runtime_config["norm_max_pts"].as<int>();
    params.sq_err_threshold = runtime_config["sq_err_threshold"].as<double>();
    params.robust_kernel_delta = runtime_config["robust_kernel_delta"].as<double>();
    const int max_iba_iter = runtime_config["max_iba_iter"].as<int>();
    const int inner_iba_iter = runtime_config["inner_iba_iter"].as<int>();
    params.verborse = runtime_config["verborse"].as<bool>();

    YAML::Node FrameIdCfg = YAML::LoadFile(KyeFrameIdFile);
    std::vector<int> vKFFrameId = FrameIdCfg["mnFrameId"].as<std::vector<int>>();
    // Pre-allocate memory to enhance peformance

    PointCloudFiles.reserve(vKFFrameId.size());
    for(auto &KFId:vKFFrameId)
    {
        PointCloudFiles.push_back(pointcloud_dir + RawPointCloudFiles[KFId]);
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
    auto solver = new g2o::OptimizationAlgorithmDogleg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(params.verborse);       // 打开调试输出
    bool multithread_flag = optimizer.initMultiThreading();
    if(!multithread_flag)
        std::cerr << "use multithread for g2o failed!" << std::endl;
    g2o::SE3Quat quat(rigid.topLeftCorner(3, 3), rigid.topRightCorner(3, 1));
    g2o::Vector7 sim3_log;
    sim3_log.head<6>() = quat.log();
    sim3_log(6) = scale;
    Eigen::Matrix3d optimizedRotation; Eigen::Vector3d optimizedTranslation; double optimizedScale;
    for(int iba_iter = 0; iba_iter < max_iba_iter; ++ iba_iter)
    {
        BuildOptimizer(PointCloudFiles, KeyFrames, sim3_log, optimizer, params);
        optimizer.initializeOptimization(0);
        optimizer.optimize(inner_iba_iter);
        const VertexSim3* v = dynamic_cast<VertexSim3*>(optimizer.vertex(0));
        sim3_log = v->estimate();
        if(params.verborse)
        {
            std::tie(optimizedRotation, optimizedTranslation, optimizedScale) = Sim3Exp<double>(sim3_log.data());
            std::cout << "IBA iter = " << iba_iter << std::endl;
            std::cout << "Rotation:\n" << optimizedRotation << std::endl;
            std::cout << "Translation: " << optimizedTranslation.transpose() << std::endl;
            std::cout << "Scale: " << optimizedScale << std::endl;
            auto logger = g2oLogEdges<IBATestEdge>(optimizer);
            print_map("Statics of Edge Error", logger);
        }
    }
    std::tie(optimizedRotation, optimizedTranslation, optimizedScale) = Sim3Exp<double>(sim3_log.data());
    std::cout << "Final Result: " << std::endl;
    std::cout << "Rotation:\n" << optimizedRotation << std::endl;
    std::cout << "Translation: " << optimizedTranslation.transpose() << std::endl;
    std::cout << "Scale: " << optimizedScale << std::endl;
    rigid.topLeftCorner(3, 3) = optimizedRotation;
    rigid.topRightCorner(3, 1) = optimizedTranslation;
    writeSim3(resFile, rigid, optimizedScale);
    std::cout << "Sim3 Result saved to " << resFile << std::endl;
    
}