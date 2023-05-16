#include <cmath>
#include <tuple>
#include <mutex>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <queue>
#include <Eigen/Dense>

#include <scancontext/Scancontext.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/pipelines/registration/PoseGraph.h>
#include <open3d/pipelines/registration/GlobalOptimization.h>
#include <open3d/pipelines/registration/GlobalOptimizationMethod.h>
#include <open3d/utility/Logging.h>
#include <open3d/io/PoseGraphIO.h>
#include <open3d/pipelines/registration/Feature.h>

#include <open3d/visualization/visualizer/Visualizer.h>

#include <kitti_tools.h>
#include <io_tools.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

gtsam::Pose3 Eigen2GTSAM(const Eigen::Isometry3d &poseEigen);
gtsam::Pose3 Eigen2GTSAM(const Eigen::Matrix4d &poseEigen);

std::tuple<double,double> PoseDif(Eigen::Isometry3d &srcPose, Eigen::Isometry3d &tgtPose);
std::tuple<double,double> PoseDif(Eigen::Matrix4d &srcPose, Eigen::Matrix4d &tgtPose);

open3d::pipelines::registration::RegistrationResult Registration(const open3d::geometry::PointCloud &srcPointCloud, const open3d::geometry::PointCloud &tgtPointCloud, const Eigen::Matrix4d &initialPose, const double icp_corr_dist);
open3d::pipelines::registration::RegistrationResult Registration(const open3d::geometry::PointCloud &srcPointCloud, const open3d::geometry::PointCloud &tgtPointCloud, const Eigen::Matrix4d &initialPose, const double icp_coarse_dist, const double icp_refine_dist);

open3d::geometry::PointCloud ReadPointCloudO3D(const std::string filename);

class BackEndOption{
public:
    bool verborse = false;
    bool vis = false;
    double voxel = 0.4;
    double o3d_voxel = 0.2;
    double keyframeMeterGap = 2;
    double keyframeRadGap = 0.2;
    double LCkeyframeMeterGap = 20;
    double LCkeyframeRadGap = 2.0;
    int LCSubmapSize = 25;

    double scDistThres = 0.2;
    double scMaximumRadius = 80;
    double loopOverlapThre = 0.8; // the overlapping area (# of inlier correspondences / # of points in target). Higher is better.
    double loopInlierRMSEThre = 0.3;
    double icp_corase_dist = 0.4 * 15;
    double icp_refine_dist = 0.4 * 1.5;
    int icp_max_iter = 30;
    
    double relinearizeThreshold = 0.01;
    int relinearizeSkip = 1;
    gtsam::ISAM2Params::Factorization factorization = gtsam::ISAM2Params::QR;
    gtsam::Vector6 priorNoiseVector6;
    gtsam::Vector6 odomNoiseVector6;
    gtsam::Vector6 robustNoiseVector6;

    double MRmaxCorrDist = 0.6;
    double MREdgePruneThre = 0.25;
    int MRmaxIter = 100;

public:
    BackEndOption(){
        priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
        odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        robustNoiseVector6 << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    }
};

class BackEndOptimizer{
public:
    BackEndOptimizer(const BackEndOption &option);
    void LoadPose(const std::string dirname);
    void LoadDataFilename(const std::string dirname);
    Eigen::Isometry3d getPose(const int i);
    Eigen::Isometry3d getrelPose(const int i, const int j);    
    open3d::geometry::PointCloud LoadPCD(const int idx) const;
    open3d::geometry::PointCloud LoadPCD(const int idx, double voxel_) const;
    open3d::geometry::PointCloud MergeLoadPCD(const int ref_idx, const int start_idx, const int end_idx);
    void AddLoopClosure(const std::tuple<int,int,open3d::geometry::PointCloud> &item);
    bool IsLoopBufEmpty();
    void AddPriorFactor(const int i, const gtsam::Pose3 &pose);
    void AddOdomFactor(const int i, const int j, const gtsam::Pose3 &relPose, const gtsam::Pose3 &currPose);
    void AddBetweenFactor(const gtsam::BetweenFactor<gtsam::Pose3> &factor);
    std::tuple<int,int,open3d::geometry::PointCloud> PopLoopClosure();
    void SCManage(const int i = 0); // for initialization
    void SCManage(const int i, const int j); // for odometry
    void LoopClosureRegThread(void);
    bool CheckStopLoopClosure(void);
    void StopLoopClosure(void);
    void UpdateISAM();
    void UpdatePosesFromEstimates(const gtsam::Values &Estimates);
    void UpdatePosesFromInitialEstimates();
    void UpdatePosesFromPG();
    void LoopClosureRun(const std::string pose_dir, const std::string file_dir);
    void PerformLoopClosure(const int j, open3d::geometry::PointCloud &PointCloud);
    void MultiRegistration(bool OdomRefinement=false);
    void writePoseGraph(const std::string &filename) const;
    void writePoses(const std::string &filename);

private:
    bool verborse;
    bool vis;
    const double voxel;
    const double o3d_voxel;
    const double keyframeMeterGap;
    const double keyframeRadGap;
    const double scDistThres;
    const double scMaximumRadius;

    const double loopOverlapThre;
    const double loopInlierRMSEThre;
    const int LCSubmapSize;
    const double LCkeyframeMeterGap;
    const double LCkeyframeRadGap;

    const double icp_corase_dist;
    const double icp_refine_dist;
    const int icp_max_iter;

    const double MRmaxCorrDist;
    const double MREdgePruneThre;
    const int MRmaxIter;

    double translationAccumulated;
    double rotationAccumulated;
    double LCtranslationAccumulated;
    double LCrotationAccumulated;

    bool IsLoopClosureStop;

    std::vector<Eigen::Isometry3d> RawFramePoses;
    std::vector<std::string> PointCloudFiles;
    SC2::SCManager SCManager;
    open3d::pipelines::registration::PoseGraph PoseGraph;

    // SCManager
    std::vector<Eigen::Isometry3d> FramePoses;
    std::vector<int> KeyFrameQuery;
    std::queue<std::tuple<int, int, open3d::geometry::PointCloud>> LoopClosureBuf;
    std::unordered_map<int, int> LoopQueryMap;
    

    // GTSAM
    Eigen::Isometry3d OdomPose;
    gtsam::ISAM2* isam;
    gtsam::Values initialEstimate;
    gtsam::NonlinearFactorGraph FactorGraph;
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise;
    gtsam::noiseModel::Diagonal::shared_ptr odomNoise;
    gtsam::noiseModel::Base::shared_ptr robustLoopNoise;

    std::mutex LoopClosureMutex;
    std::mutex LoopBufMutex;
    std::mutex FactorGraphMutex;
    std::mutex FramePoseMutex;


};