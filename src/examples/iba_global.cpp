#include "KDTreeVectorOfVectorsAdaptor.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o_tools.h"
#include "io_tools.h"
#include "kitti_tools.h"
#include "pointcloud.h"
#include <mutex>
#include <functional>
#include <limits>
#include <mutex>
#include <yaml-cpp/yaml.h>
#include "orb_slam/include/System.h"
#include "orb_slam/include/KeyFrame.h"
#include <opencv2/core/eigen.hpp>
#include "Nomad/nomad.hpp"
#include "Cache/CacheBase.hpp"


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
    int best_convis_num = 3;
    double err_threshold = 400.;
    std::vector<double> lb;
    std::vector<double> ub;
    int PointCloudskip = 1;
    int max_bbeval = 200;
    bool verborse = true;
    int curr_eval = 0;
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

std::tuple<double, int, int> BAError(double* xvec, const std::vector<VecVector3d> &PointClouds, const std::vector<ORB_SLAM2::KeyFrame*> &KeyFrames, const IBAGlobalParams &iba_params)
{
    double fval = 0.; // total error
    int edge_cnt = 0;
    int valid_edge_cnt = 0;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double scale;
    std::tie(rotation, translation, scale) = Sim3Exp<double>(xvec);
    Eigen::Isometry3d transformation(Eigen::Isometry3d::Identity());
    transformation.rotate(rotation);
    transformation.pretranslate(translation);
    for(std::size_t i = 0; i < PointClouds.size(); ++i)
    {
        VecVector3d points = PointClouds[i];
        VecVector3d normals;
        const ORB_SLAM2::KeyFrame* pKF = KeyFrames[i];
        const double H = pKF->mnMaxY, W = pKF->mnMaxX;
        TransformPointCloudInplace(points, transformation); // Transfer to Camera coordinate
        CorrSet corrset; // pair of (idx of image points, idx of pointcloud points)
        FindProjectCorrespondences(points, pKF, iba_params.kdtree2d_max_leaf_size, iba_params.max_pixel_dist, corrset);
        if(corrset.size() < 50)  // too few correspondences
            continue;
        std::vector<ORB_SLAM2::KeyFrame*> ConvisKeyFrames = pKF->GetBestCovisibilityKeyFramesSafe(iba_params.best_convis_num);  
        std::vector<std::map<int, int>> KptMapList; // Keypoint-Keypoint Corr
        std::vector<Eigen::Matrix4d> relPoseList; // RelPose From Reference to Convisible KeyFrames
        KptMapList.reserve(ConvisKeyFrames.size());
        relPoseList.reserve(ConvisKeyFrames.size());
        const cv::Mat invRefPose = pKF->GetPoseInverseSafe();
        for(auto pKFConv:ConvisKeyFrames)
        {
            auto KptMap = pKF->GetMatchedKptIds(pKFConv);
            cv::Mat relPose = pKFConv->GetPose() * invRefPose;  // Transfer from c1 coordinate to c2 coordinate
            Eigen::Matrix4d relPoseEigen;
            cv::cv2eigen(relPose, relPoseEigen);
            relPoseEigen.topRightCorner(3, 1) *= scale; // noalias
            KptMapList.push_back(std::move(KptMap));
            relPoseList.push_back(std::move(relPoseEigen));
        }
        
        for(auto &corr:corrset)
        {
            const int point2d_idx = corr.first;  // KeyPoint Idx matched with PointCloud
            const int point3d_idx = corr.second; // Point Idx matched with KeyPoints
            double u0 = pKF->mvKeysUn[point2d_idx].pt.x;
            double v0 = pKF->mvKeysUn[point2d_idx].pt.y;
            Eigen::Vector3d p0 = points[point3d_idx]; // camera coord
            const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
            // transform 3d point back to LiDAR coordinate
            for(std::size_t pKFConvi = 0; pKFConvi < ConvisKeyFrames.size(); ++pKFConvi){
                auto pKFConv = ConvisKeyFrames[pKFConvi];
                // Skip if Cannot Find this 2d-3d matching map in Keypoint-to-Keypoint matching map
                if(KptMapList[pKFConvi].count(point2d_idx) == 0)
                    continue;
                const int convis_idx = KptMapList[pKFConvi][point2d_idx];  // corresponding KeyPoints idx in a convisible KeyFrame
                double u1 = pKFConv->mvKeysUn[convis_idx].pt.x;
                double v1 = pKFConv->mvKeysUn[convis_idx].pt.y;
                Eigen::Matrix4d relPose = relPoseList[pKFConvi];  // Twc2 * inv(Twc1)
                Eigen::Vector3d p1 = relPose.topLeftCorner(3, 3) * p0 + relPose.topRightCorner(3, 1); // transform to covisible Keyframe coord
                double obs_u1 = fx * p1[0]/p1[2] + cx;
                double obs_v1 = fy * p1[1]/p1[2] + cy;
                // printf("u0: %lf, v0: %lf, u1:%lf, v1:%lf, obs_u1:%lf, obs_v1:%lf\n", u0, v0, u1, v1, obs_u1, obs_v1);
                double err = (obs_u1 - u1) * (obs_u1 - u1) + (obs_v1 - v1) * (obs_v1 - v1);
                err = sqrt(err);
                if(err < iba_params.err_threshold)
                {
                    fval += err;
                    valid_edge_cnt++;
                }
                edge_cnt++;
            }
        }
    }
    return {fval, valid_edge_cnt, edge_cnt};
}

class BALoss: public NOMAD::Evaluator
{
public:
    BALoss(const std::shared_ptr<NOMAD::EvalParameters>& evalParams, const std::vector<std::string> &_PointCloudFiles, 
        const std::vector<ORB_SLAM2::KeyFrame*>& _KeyFrames, const IBAGlobalParams &_iba_params):
        NOMAD::Evaluator(evalParams, NOMAD::EvalType::BB), KeyFrames(_KeyFrames), iba_params(_iba_params)
        {
            PointClouds.reserve(_PointCloudFiles.size());
            for(std::size_t i = 0; i < _PointCloudFiles.size(); ++i)
            {
                VecVector3d PointCloud;
                readPointCloud(_PointCloudFiles[i], PointCloud);
                PointClouds.push_back(std::move(PointCloud));
                if(iba_params.verborse && i % 100 == 0)
                    std::printf("Read %0.2lf %% PointClouds\n", 100.0*(i+1)/_PointCloudFiles.size());
            }
        }
    ~BALoss(){}
    bool eval_x(NOMAD::EvalPoint &x, const NOMAD::Double &hMax, bool &countEval) const override
    {
        double fval = 0.; // total error
        int edge_cnt = 0;
        int valid_edge_cnt = 0;
        double xvec[7];
        for(int i = 0; i < 7; ++i)
            xvec[i] = x[i].todouble();
        std::tie(fval, valid_edge_cnt, edge_cnt) = BAError(xvec, PointClouds, KeyFrames, iba_params);
        NOMAD::Double f = fval / valid_edge_cnt;
        std::string bbo = f.tostring();
        x.setBBO(bbo);
        countEval = true;
        return true;
    }

private:
    std::vector<VecVector3d> PointClouds;
    std::vector<ORB_SLAM2::KeyFrame*> KeyFrames;
    IBAGlobalParams iba_params;
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
    // ORB Config
    const std::string ORBVocFile = orb_config["Vocabulary"].as<std::string>();
    const std::string ORBCfgFile = orb_config["Config"].as<std::string>();
    std::string ORBKeyFrameDir = orb_config["KeyFrameDir"].as<std::string>();
    checkpath(ORBKeyFrameDir);
    const std::string ORBMapFile = orb_config["MapFile"].as<std::string>();
    // runtime config
    iba_params.max_pixel_dist = runtime_config["max_pixel_dist"].as<double>();
    iba_params.best_convis_num = runtime_config["best_convis_num"].as<int>();
    iba_params.kdtree2d_max_leaf_size = runtime_config["kdtree2d_max_leaf_size"].as<int>();
    iba_params.kdtree3d_max_leaf_size = runtime_config["kdtree3d_max_leaf_size"].as<int>();
    iba_params.err_threshold = runtime_config["err_threshold"].as<double>();
    iba_params.lb = runtime_config["lb"].as<std::vector<double>>();
    iba_params.ub = runtime_config["ub"].as<std::vector<double>>();
    iba_params.PointCloudskip = io_config["PointCloudskip"].as<int>();
    iba_params.max_bbeval = runtime_config["max_bbeval"].as<int>();
    iba_params.verborse = runtime_config["verborse"].as<bool>();
    const std::string NomadCacheFile = runtime_config["cacheFile"].as<std::string>();
    const double mesh_precision = runtime_config["mesh_precision"].as<double>();
    std::vector<double> mesh_precision_arr(7, mesh_precision);
    YAML::Node FrameIdCfg = YAML::LoadFile(KyeFrameIdFile);
    std::vector<int> vKFFrameId = FrameIdCfg["mnFrameId"].as<std::vector<int>>();
    // Pre-allocate memory to enhance peformance
    PointCloudFiles.reserve(vKFFrameId.size());
    for(auto &KFId:vKFFrameId)
        PointCloudFiles.push_back(pointcloud_dir + RawPointCloudFiles[KFId]);
    RawPointCloudFiles.clear(); 
    ORB_SLAM2::System orb_slam(ORBVocFile, ORBCfgFile, ORB_SLAM2::System::MONOCULAR, false);
    orb_slam.Shutdown(); // Do not need any ORB running threads
    orb_slam.RestoreSystemFromFile(ORBKeyFrameDir, ORBMapFile);
    KeyFrames = orb_slam.GetAllKeyFrames(true);
    std::sort(KeyFrames.begin(), KeyFrames.end(), ORB_SLAM2::KeyFrame::lId);
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
    std::vector<VecVector3d> PointClouds;
    PointClouds.reserve(PointCloudFiles.size());
    // double fval;
    // int valid_edge_cnt;
    // int edge_cnt;
    // std::cout << "Eval specific points\n";
    // std::tie(fval, valid_edge_cnt, edge_cnt) = BAError(x0.data(), PointClouds, KeyFrames, iba_params);
    // printf("BA Loss at init_x0: %lf\n", fval / valid_edge_cnt);
    // std::tie(fval, valid_edge_cnt, edge_cnt) = BAError(gtx0.data(), PointClouds, KeyFrames, iba_params);
    // std::printf("BA Loss at gt_x0: %lf\n", fval / valid_edge_cnt);

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
    
    // Set NOMAD Parameters
    auto nomad_params = std::make_shared<NOMAD::AllParameters>();
    nomad_params->set_MAX_BB_EVAL(iba_params.max_bbeval);
    nomad_params->set_DIMENSION(7);
    NOMAD::BBOutputTypeList bbOutputTypes;
    bbOutputTypes.push_back(NOMAD::BBOutputType::OBJ);
    nomad_params->set_BB_OUTPUT_TYPE(bbOutputTypes);
    NOMAD::ArrayOfDouble nomad_lb(lb), nomad_ub(ub);
    nomad_params->set_LOWER_BOUND(nomad_lb);
    nomad_params->set_UPPER_BOUND(nomad_ub);
    NOMAD::Point nomad_x0(x0); NOMAD::Point nomad_gtx0(gtx0);
    nomad_params->set_X0(nomad_x0);
    nomad_params->setAttributeValue("DISPLAY_STATS", NOMAD::ArrayOfString("BBE ( SOL ) OBJ"));
    nomad_params->setAttributeValue("CACHE_FILE", NomadCacheFile);
    NOMAD::ArrayOfDouble nomad_mesh_minsize(mesh_precision_arr);
    nomad_params->set_MIN_MESH_SIZE(nomad_mesh_minsize);
    nomad_params->checkAndComply();
    auto ev = std::make_unique<BALoss>(nomad_params->getEvalParams(), PointCloudFiles, KeyFrames, iba_params);
    NOMAD::MainStep nomad_main_step;
    nomad_main_step.setAllParameters(nomad_params);
    nomad_main_step.setEvaluator(std::move(ev));
    nomad_main_step.start();
    std::cout << "NOMAD Start\n";
    nomad_main_step.run();
    nomad_main_step.end();
    std::vector<NOMAD::EvalPoint> evalPointFeasList;
    auto nbFeas = NOMAD::CacheBase::getInstance()->findBestFeas(evalPointFeasList, NOMAD::Point(), 
        NOMAD::EvalType::BB, NOMAD::ComputeType::STANDARD, nullptr);
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
    writeSim3(resFile, rigid, optScale);
    return 0;

}