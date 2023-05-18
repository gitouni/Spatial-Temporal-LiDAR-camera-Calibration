#include "backend_opt.h"


inline gtsam::Pose3 Eigen2GTSAM(const Eigen::Isometry3d &poseEigen){
    return gtsam::Pose3(poseEigen.matrix() );
}

inline gtsam::Pose3 Eigen2GTSAM(const Eigen::Matrix4d &poseEigen){
    return gtsam::Pose3(poseEigen);
}

std::tuple<double,double> PoseDif(Eigen::Isometry3d &srcPose, Eigen::Isometry3d &tgtPose){
    double delta_translation = (tgtPose.translation()- srcPose.translation()).norm();
    Eigen::AngleAxisd ax(tgtPose.rotation() * srcPose.rotation().transpose());
    double delta_rotation = ax.angle();
    return {delta_translation, delta_rotation};
}

std::tuple<double,double> PoseDif(Eigen::Matrix4d &srcPose, Eigen::Matrix4d &tgtPose){
    double delta_translation = (tgtPose.block<3, 1>(0, 3)- srcPose.block<3, 1>(0, 3)).norm();
    Eigen::AngleAxisd ax(tgtPose.block<3, 3>(0, 0) * srcPose.block<3, 3>(0, 0).transpose());
    double delta_rotation = ax.angle();
    return {delta_translation, delta_rotation};
}


open3d::pipelines::registration::RegistrationResult Registration(const open3d::geometry::PointCloud &srcPointCloud, const open3d::geometry::PointCloud &tgtPointCloud, const Eigen::Matrix4d &initialTran, const double icp_corr_dist){
    open3d::pipelines::registration::RegistrationResult reg = open3d::pipelines::registration::RegistrationICP(
        srcPointCloud, tgtPointCloud, icp_corr_dist, initialTran,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6, icp_corr_dist));
    return reg;
}

open3d::pipelines::registration::RegistrationResult Registration(const open3d::geometry::PointCloud &srcPointCloud, const open3d::geometry::PointCloud &tgtPointCloud, const Eigen::Matrix4d &initialTran, const double icp_coarse_dist, const double icp_refine_dist){
    open3d::pipelines::registration::RegistrationResult reg_coarse = open3d::pipelines::registration::RegistrationICP(
        srcPointCloud, tgtPointCloud, icp_coarse_dist, initialTran,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-4,1e-4, icp_coarse_dist));
    open3d::pipelines::registration::RegistrationResult reg = open3d::pipelines::registration::RegistrationICP(
        srcPointCloud, tgtPointCloud, icp_refine_dist, reg_coarse.transformation_,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6, icp_refine_dist));
    return reg;
}

open3d::pipelines::registration::RegistrationResult RANSACRegistration(open3d::geometry::PointCloud &srcPointCloud, open3d::geometry::PointCloud &tgtPointCloud, double max_corr_dist){
    if(!srcPointCloud.HasNormals())
        srcPointCloud.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(1.2,30));
    if(!tgtPointCloud.HasNormals())
        tgtPointCloud.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(1.2,30));
    auto srcFeature = open3d::pipelines::registration::ComputeFPFHFeature(srcPointCloud,
        open3d::geometry::KDTreeSearchParamHybrid(5,100));
    auto tgtFeature = open3d::pipelines::registration::ComputeFPFHFeature(tgtPointCloud,
        open3d::geometry::KDTreeSearchParamHybrid(5,100));
    open3d::pipelines::registration::RegistrationResult reg = open3d::pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
        srcPointCloud, tgtPointCloud, *srcFeature, *tgtFeature, false, max_corr_dist, 
        open3d::pipelines::registration::TransformationEstimationPointToPoint()
    );

    return reg;
}

open3d::pipelines::registration::RegistrationResult PLRegistration(const open3d::geometry::PointCloud &srcPointCloud, const open3d::geometry::PointCloud &tgtPointCloud, const Eigen::Matrix4d &initialTran, const double icp_corr_dist){
    open3d::pipelines::registration::RegistrationResult reg = open3d::pipelines::registration::RegistrationICP(
        srcPointCloud, tgtPointCloud, icp_corr_dist, initialTran,
        open3d::pipelines::registration::TransformationEstimationPointToPlane(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6, icp_corr_dist));
    return reg;
}

open3d::pipelines::registration::RegistrationResult PLRegistration(const open3d::geometry::PointCloud &srcPointCloud, const open3d::geometry::PointCloud &tgtPointCloud, const Eigen::Matrix4d &initialTran, const double icp_coarse_dist, const double icp_refine_dist){
    open3d::pipelines::registration::RegistrationResult reg_coarse = open3d::pipelines::registration::RegistrationICP(
        srcPointCloud, tgtPointCloud, icp_coarse_dist, initialTran,
        open3d::pipelines::registration::TransformationEstimationPointToPlane(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-4,1e-4, icp_coarse_dist));
    open3d::pipelines::registration::RegistrationResult reg = open3d::pipelines::registration::RegistrationICP(
        srcPointCloud, tgtPointCloud, icp_refine_dist, reg_coarse.transformation_,
        open3d::pipelines::registration::TransformationEstimationPointToPlane(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6, icp_refine_dist));
    return reg;
}

open3d::geometry::PointCloud ReadPointCloudO3D(const std::string filename){
    std::vector<Eigen::Vector3d> PointCloudEigen;
    readPointCloud(filename, PointCloudEigen);
    return open3d::geometry::PointCloud(PointCloudEigen);
}

BackEndOptimizer::BackEndOptimizer(const BackEndOption &option):
    verborse(option.verborse),vis(option.vis),voxel(option.voxel),o3d_voxel(o3d_voxel),
    keyframeMeterGap(option.keyframeMeterGap),keyframeRadGap(option.keyframeRadGap),
    scDistThres(option.scDistThres),scMaximumRadius(option.scMaximumRadius),
    loopOverlapThre(option.loopOverlapThre),loopInlierRMSEThre(option.loopInlierRMSEThre),
    LCSubmapSize(option.LCSubmapSize),LCkeyframeMeterGap(option.LCkeyframeMeterGap),LCkeyframeRadGap(option.LCkeyframeRadGap),
    icp_corase_dist(option.icp_corase_dist),icp_refine_dist(option.icp_refine_dist),icp_max_iter(option.icp_max_iter),
    MRmaxCorrDist(option.MRmaxCorrDist),MREdgePruneThre(option.MREdgePruneThre),MRmaxIter(option.MRmaxIter),
    translationAccumulated(0),rotationAccumulated(0),LCtranslationAccumulated(0),LCrotationAccumulated(0),
    IsLoopClosureStop(false),
    OdomPose(Eigen::Isometry3d::Identity())
    {
        gtsam::ISAM2Params isam_params;
        isam_params.factorization = option.factorization;
        isam_params.relinearizeSkip = option.relinearizeSkip;
        isam_params.relinearizeThreshold = option.relinearizeThreshold;
        isam = new gtsam::ISAM2(isam_params);

        priorNoise = gtsam::noiseModel::Diagonal::Variances(option.priorNoiseVector6);
        odomNoise = gtsam::noiseModel::Diagonal::Variances(option.odomNoiseVector6);
        robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // alternatives: DCS or GemanMcClure
                    gtsam::noiseModel::Diagonal::Variances(option.robustNoiseVector6)
                    );
    }

void BackEndOptimizer::LoadPose(const std::string filename){
    ReadPoseList(filename, RawFramePoses); 
    FramePoses.resize(RawFramePoses.size());
    std::copy(RawFramePoses.begin(), RawFramePoses.end(), FramePoses.begin());
    OdomPose = RawFramePoses[0];
    if(verborse)
        std::cout << "Load " << FramePoses.size() << " Poses." << std::endl; 
}

void BackEndOptimizer::LoadDataFilename(const std::string dirname_){
    std::string dirname(dirname_);
    checkpath(dirname);
    listdir(PointCloudFiles, dirname);
    for(std::string &filename:PointCloudFiles)
        filename = dirname + filename;
    if(verborse)
        std::cout << "Load " << PointCloudFiles.size() << " PointCloud Filenames." << std::endl; 
}

void BackEndOptimizer::writePoseGraph(const std::string &filename) const {
    bool res = open3d::io::WritePoseGraph(filename, PoseGraph);
    if(verborse){
        if(res)
            std::cout << "\033[32;1m" << "PoseGraph has been successfully written to " << filename << ".\033[0m" << std::endl;
        else
            std::cout << "\033[33;1m" << "PoseGraph FAILTED TO BE written to " << filename << "!\033[0m" << std::endl;
    }
}

void BackEndOptimizer::writePoses(const std::string &filename) {
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    std::ofstream of(filename);
    for(int i = 0; i < (int)FramePoses.size(); ++i){
        writeKittiData(of, FramePoses[i], i == FramePoses.size()-1);
    }
    of.close();
}

Eigen::Isometry3d BackEndOptimizer::getPose(const int i){
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    return FramePoses[i];
}
Eigen::Isometry3d BackEndOptimizer::getrelPose(const int i, const int j){
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    return FramePoses[j] * FramePoses[i].inverse();
}


open3d::geometry::PointCloud BackEndOptimizer::LoadPCD(const int idx) const{
    open3d::geometry::PointCloud OutputPointCloud = ReadPointCloudO3D(PointCloudFiles[idx]);
    return *(OutputPointCloud.VoxelDownSample(voxel));
}

open3d::geometry::PointCloud BackEndOptimizer::LoadPCD(const int idx, double voxel_) const{
    open3d::geometry::PointCloud OutputPointCloud = ReadPointCloudO3D(PointCloudFiles[idx]);
    return *(OutputPointCloud.VoxelDownSample(voxel_));
}

open3d::geometry::PointCloud BackEndOptimizer::MergeLoadPCD(const int ref_idx, const int start_idx, const int end_idx){
    open3d::geometry::PointCloud tmpOutputPointCloud;
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    for(int i=start_idx; i<end_idx; ++i){
        open3d::geometry::PointCloud tmpPointCloud = ReadPointCloudO3D(PointCloudFiles[i]);
        tmpPointCloud.Transform(FramePoses[i].matrix());
        tmpOutputPointCloud += tmpPointCloud;
    }
    open3d::geometry::PointCloud OutputPointCloud = *(tmpOutputPointCloud.VoxelDownSample(voxel));
    OutputPointCloud.Transform(FramePoses[ref_idx].inverse().matrix());
    return OutputPointCloud;
}


void BackEndOptimizer::AddLoopClosure(const std::tuple<int,int,open3d::geometry::PointCloud> &item){
    std::unique_lock<std::mutex> lock(LoopBufMutex);
    LoopClosureBuf.push(item);
}

bool BackEndOptimizer::IsLoopBufEmpty(){
    std::unique_lock<std::mutex> lock(LoopBufMutex);
    return LoopClosureBuf.empty();
}

void BackEndOptimizer::AddPriorFactor(const int i, const gtsam::Pose3 &pose){
    std::unique_lock<std::mutex> lock(FactorGraphMutex);
    FactorGraph.add(gtsam::PriorFactor<gtsam::Pose3>(i, pose, priorNoise));
    initialEstimate.insert(i, pose);
}

void BackEndOptimizer::AddOdomFactor(const int i, const int j, const gtsam::Pose3 &prevPose, const gtsam::Pose3 &currPose){
    std::unique_lock<std::mutex> lock(FactorGraphMutex);
    FactorGraph.add(gtsam::BetweenFactor<gtsam::Pose3>(i, j,  prevPose.between(currPose), odomNoise));
    initialEstimate.insert(j, currPose);
}

void BackEndOptimizer::AddBetweenFactor(const gtsam::BetweenFactor<gtsam::Pose3> &factor){
    std::unique_lock<std::mutex> lock(FactorGraphMutex);
    FactorGraph.add(factor);
}

std::tuple<int,int,open3d::geometry::PointCloud> BackEndOptimizer::PopLoopClosure(void){
    std::unique_lock<std::mutex> lock(LoopBufMutex);
    auto item = LoopClosureBuf.front();
    LoopClosureBuf.pop();
    return item;
}



void BackEndOptimizer::StopLoopClosure(void){
    {
        std::unique_lock<std::mutex> lock(LoopClosureMutex);
        IsLoopClosureStop = true;
    }
}

bool BackEndOptimizer::CheckStopLoopClosure(void){
    {
        std::unique_lock<std::mutex> lock(LoopClosureMutex);
        return IsLoopClosureStop;
    }
}

/**
 * @brief Loop Closure ICP (Used for Thread)
 * , use StopLoopClosure() to stop this thread
 */
void BackEndOptimizer::LoopClosureRegThread(void){
    std::cout << "\033[32;1m" << "Loop CLosure Thread Started." << "\033[0m" << std::endl; 
    while(1)
    {
        if(IsLoopBufEmpty()){
            if(!CheckStopLoopClosure()){
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            else
                break;
        }
        int history_frame_id, curr_frame_id;
        open3d::geometry::PointCloud PointCloud;
        std::tie(history_frame_id, curr_frame_id, PointCloud) = PopLoopClosure();
        if(verborse)
            std::cout << "Perform ICP between " << history_frame_id << " & " << curr_frame_id << std::endl; 
        int start_idx = std::max(0, history_frame_id-LCSubmapSize);
        int end_idx = std::min(history_frame_id+LCSubmapSize, (int)PointCloudFiles.size());
        auto HistoryPointCloud = MergeLoadPCD(history_frame_id, start_idx, end_idx);
            auto reg = Registration(PointCloud, HistoryPointCloud, Eigen::Matrix4d::Identity(), icp_corase_dist, icp_refine_dist);  // Untransformed PointClouds
            if(reg.fitness_ > loopOverlapThre && reg.inlier_rmse_ < loopInlierRMSEThre) // Valid Loop Closure
            {
                
                AddBetweenFactor(gtsam::BetweenFactor<gtsam::Pose3>(history_frame_id, curr_frame_id, Eigen2GTSAM(reg.transformation_), robustLoopNoise));  // GTSAM use inverse Between Pose
                LoopQueryMap.insert(std::make_pair(history_frame_id, curr_frame_id));
                if(verborse){
                    char msg[100];
                    sprintf(msg, "Loop Closure Valid: %d & %d, RMSE: %lf, Overlap: %lf", history_frame_id, curr_frame_id, reg.inlier_rmse_, reg.fitness_);
                    std::cout << "\033[32;1m" << msg << "\033[0m" << std::endl;
                }
                UpdateISAM();
            }else
            {
                if(verborse){
                    char msg[100];
                    sprintf(msg, "Loop Closure Invalid: %d & %d, RMSE: %lf, Overlap: %lf", history_frame_id, curr_frame_id, reg.inlier_rmse_, reg.fitness_);
                    std::cout << "\033[31;1m" << msg << "\033[0m" << std::endl;
                }
            }
            if(vis){
                PointCloud.PaintUniformColor((Eigen::Vector3d() << 0.2, 0.5, 0.8).finished());
                HistoryPointCloud.PaintUniformColor((Eigen::Vector3d() << 0.7, 0.9, 0.2).finished());
                open3d::visualization::Visualizer visualizer;
                visualizer.CreateVisualizerWindow("Open3D Visualizer", 640, 480);
                std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr(new open3d::geometry::PointCloud(PointCloud.Transform(reg.transformation_)));
                std::shared_ptr<open3d::geometry::PointCloud> HistoryPointCloudPtr(new open3d::geometry::PointCloud(HistoryPointCloud));
                visualizer.AddGeometry(PointCloudPtr);
                visualizer.AddGeometry(HistoryPointCloudPtr);
                visualizer.Run();
            }
    }
    if(verborse)
        std::cout << "\033[32;1m" << "Loop CLosure Thread Stopped Successfully." << "\033[0m" << std::endl; 
}

/**
 * @brief Perform Loop Closure ICP and ISAM Update to reduce accumulated error
 * 
 * @param j current node index
 * @param PointCloud current PointCloud
 */
void BackEndOptimizer::PerformLoopClosure(const int j, open3d::geometry::PointCloud &PointCloud){
    auto detectResult = SCManager.detectLoopClosureID();
    int history_node_id = detectResult.first;
    if(history_node_id != -1)  // Loop Closure Detected
    {
        int history_frame_id = KeyFrameQuery[history_node_id];
        AddLoopClosure(std::make_tuple(history_frame_id, j, PointCloud));
        if(verborse)
            std::cout << "Loop Closure Detect: " << history_frame_id << " & " << j << std::endl;
    }
}

/**
 * @brief SCManage for initialization
 * 
 * @param PointCloud the first Frame
 * @param i default: 0
 */
void BackEndOptimizer::SCManage(const int i){
    assert(0<=i && i< (int) PointCloudFiles.size());
    auto PointCloud = LoadPCD(i);
    SCManager.makeAndSaveScancontextAndKeys(PointCloud.points_);
    auto OdomGTSAM = Eigen2GTSAM(OdomPose);
    AddPriorFactor(i, OdomGTSAM);
    KeyFrameQuery.push_back(i);
}

/**
 * @brief SCManage for Odometry
 * 
 * @param PointCloud New Frame
 * @param i previouse node index
 * @param j current node index (supposed to be i+1)
 */
void BackEndOptimizer::SCManage(const int i, const int j){
    assert(0<=i && i< (int) PointCloudFiles.size());
    assert(0<=j && j< (int) PointCloudFiles.size() && i < j);
    Eigen::Isometry3d RelPose = getrelPose(i, j);
    Eigen::Isometry3d CurrPose = getPose(j);
    AddOdomFactor(i, j, Eigen2GTSAM(OdomPose), Eigen2GTSAM(CurrPose));
    double translationDif = 0, rotationDif = 0;
    std::tie(translationDif, rotationDif) = PoseDif(OdomPose, CurrPose);
    OdomPose = CurrPose;  // Previous Pose
    if(translationDif > 5)
        std::cout << "Abnormal Rel Pose Between" << i << " & " << j << std::endl;
    translationAccumulated += translationDif;
    rotationAccumulated += rotationDif;
    LCtranslationAccumulated += translationDif;
    LCrotationAccumulated += rotationDif;
    // if(verborse)
    //     std::cout << "Frame " << j << " rotationDif: " << rotationDif << " translation Dif: " << translationDif << std::endl;
    if(translationAccumulated > keyframeMeterGap || rotationAccumulated > keyframeRadGap)  // Add KeyFrame
    {
        translationAccumulated = 0;
        rotationAccumulated = 0;
        KeyFrameQuery.push_back(j);
        auto PointCloud = LoadPCD(i);
        SCManager.makeAndSaveScancontextAndKeys(PointCloud.points_);
        if(LCtranslationAccumulated > LCkeyframeMeterGap || LCrotationAccumulated > LCkeyframeRadGap) // Perform LoopClosure
        {
            LCtranslationAccumulated = 0;
            LCrotationAccumulated = 0;
            PerformLoopClosure(j, PointCloud);
            // UpdateISAM();
        }
    }
}

 void BackEndOptimizer::UpdateISAM(){
    {
        std::unique_lock<std::mutex> lock(FactorGraphMutex);
        isam->update(FactorGraph, initialEstimate);
        isam->update();
        FactorGraph.resize(0);
        initialEstimate.clear();
    }
    gtsam::Values isamCurrentEstimate = isam->calculateBestEstimate();
    UpdatePosesFromEstimates(isamCurrentEstimate);
 }

void BackEndOptimizer::UpdatePosesFromInitialEstimates(){
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    for(int i = 0; i < (int) initialEstimate.size(); ++i){
        FramePoses[i].matrix() = initialEstimate.at<gtsam::Pose3>(i).matrix();
    }
    if(verborse)
        std::cout << "Poses Updated to "  << initialEstimate.size() << std::endl;
}

void BackEndOptimizer::UpdatePosesFromEstimates(const gtsam::Values &Estimates){
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    for(int i = 0; i < (int) Estimates.size(); ++i){
        FramePoses[i].matrix() = Estimates.at<gtsam::Pose3>(i).matrix();
    }
    if(verborse)
        std::cout << "Poses Updated to "  << Estimates.size() << std::endl;
}

void BackEndOptimizer::UpdatePosesFromPG(){
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    for(int i = 0; i < (int)PoseGraph.nodes_.size(); ++i){
        FramePoses[i].matrix() = PoseGraph.nodes_[i].pose_.inverse();
    }
    if(verborse)
        std::cout << "" << PoseGraph.nodes_.size() << " Poses Updated from PoseGraph" << std::endl;
}

/**
 * @brief Auto Run the Loop Closure Optimization
 * 
 * @param pose_dir Diranme of Pose Files
 * @param file_dir Dirname of PointCloud Files
 */
void BackEndOptimizer::LoopClosureRun(const std::string pose_dir, const std::string file_dir){
    LoadPose(pose_dir);
    LoadDataFilename(file_dir);
    assert(FramePoses.size()==PointCloudFiles.size());
    for(int i = 0; i < (int)FramePoses.size(); ++i){
        if(i == 0) // initialization
        {
            SCManage(i);
            continue;
        }
        SCManage(i-1, i);
    }
    if(verborse)
        std::cout << "LoopClosure Optimization Finished." << std::endl;
}

void BackEndOptimizer::MultiRegistration(bool OdomRefinement){
    assert(FramePoses.size() == PointCloudFiles.size() && FramePoses.size() > 0);
    if(verborse){
        char msg[100];
        sprintf(msg, "Start Multiway Registration with %ld Nodes and %ld Valid Loop Closures", FramePoses.size(), LoopQueryMap.size());
        std::cout << "\033[32;1m" << msg << "\033[0m" << std::endl;
    }
    for(Eigen::Isometry3d &pose:FramePoses){
        open3d::pipelines::registration::PoseGraphNode node(pose.inverse().matrix()); // np.linalg.inv(odometry)
        PoseGraph.nodes_.push_back(node);
    }
    if(verborse)
        std::cout << "" << PoseGraph.nodes_.size() << " Nodes Added." << std::endl;
    if(OdomRefinement){
        open3d::geometry::PointCloud lastPointCloud = LoadPCD(0);
        for(int i = 1; i < (int)FramePoses.size(); ++ i){
            open3d::geometry::PointCloud CurrPointCloud = LoadPCD(i);
            Eigen::Matrix4d relPose = getrelPose(i-1, i).matrix();
            auto reg = Registration(lastPointCloud, CurrPointCloud, relPose, icp_refine_dist);
            Eigen::Matrix6d info = open3d::pipelines::registration::GetInformationMatrixFromPointClouds(
                lastPointCloud, CurrPointCloud, MRmaxCorrDist, reg.transformation_);
            open3d::pipelines::registration::PoseGraphEdge edge(
                i-1, i, reg.transformation_, info
            );
            PoseGraph.edges_.push_back(edge);
            lastPointCloud = CurrPointCloud;
            if(verborse && (i % 100 == 0)){
                char msg[100];
                sprintf(msg, "Odom Refinement: %0.2lf %% Finished", 100.0*i/(FramePoses.size()-1));
                std::cout << msg << std::endl;
            }
        }
    }else{
        open3d::geometry::PointCloud lastPointCloud = LoadPCD(0);
        for(int i = 1; i < (int)FramePoses.size(); ++ i){
            open3d::geometry::PointCloud CurrPointCloud = LoadPCD(i);
            Eigen::Matrix4d relPose = getrelPose(i-1, i).matrix();
            Eigen::Matrix6d info = open3d::pipelines::registration::GetInformationMatrixFromPointClouds(
                lastPointCloud, CurrPointCloud, MRmaxCorrDist, relPose);
            open3d::pipelines::registration::PoseGraphEdge edge(
                i-1, i,relPose, info
            );
            PoseGraph.edges_.push_back(edge);
            lastPointCloud = CurrPointCloud;
            if(verborse && (i % 100 == 0)){
                char msg[100];
                sprintf(msg, "Odom Copied: %0.2lf %% Finished", 100.0*i/(FramePoses.size()-1));
                std::cout << msg << std::endl;
            }
        }
    }
    
    if(verborse)
        std::cout << "Odometry: " << PoseGraph.edges_.size() << " edges Added." << std::endl;
    for(auto &item:LoopQueryMap){
        open3d::geometry::PointCloud srcPointCloud = LoadPCD(item.first);
        open3d::geometry::PointCloud tgtPointCloud = LoadPCD(item.second);
        // srcPointCloud.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(o3d_voxel*3,30));
        // tgtPointCloud.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(o3d_voxel*3,30));
        Eigen::Matrix4d relPose = getrelPose(item.first, item.second).matrix();
        auto reg = Registration(srcPointCloud, tgtPointCloud, relPose, icp_refine_dist);
        if(vis){
            srcPointCloud.PaintUniformColor((Eigen::Vector3d() << 0.2, 0.5, 0.8).finished());
            tgtPointCloud.PaintUniformColor((Eigen::Vector3d() << 0.7, 0.9, 0.2).finished());
            open3d::visualization::Visualizer visualizer;
            visualizer.CreateVisualizerWindow("Open3D Visualizer", 640, 480);
            std::shared_ptr<open3d::geometry::PointCloud> srcPointCloudPtr(new open3d::geometry::PointCloud(srcPointCloud.Transform(reg.transformation_)));
            std::shared_ptr<open3d::geometry::PointCloud> tgtPointCloudPtr(new open3d::geometry::PointCloud(tgtPointCloud));
            visualizer.AddGeometry(srcPointCloudPtr);
            visualizer.AddGeometry(tgtPointCloudPtr);
            visualizer.Run();
        }
        Eigen::Matrix6d info = open3d::pipelines::registration::GetInformationMatrixFromPointClouds(
            srcPointCloud, tgtPointCloud, MRmaxCorrDist, reg.transformation_);
         open3d::pipelines::registration::PoseGraphEdge edge(
            item.first, item.second, reg.transformation_, info, true
        );
        PoseGraph.edges_.push_back(edge);
    }
    if(verborse)
        std::cout << "Loop Closure: " << LoopQueryMap.size() << " edges Added." << std::endl;
    // Global Registration
    auto option = open3d::pipelines::registration::GlobalOptimizationOption(
        MRmaxCorrDist, MREdgePruneThre, 1.0, 0
    );
    auto criteria = open3d::pipelines::registration::GlobalOptimizationConvergenceCriteria();
    criteria.max_iteration_ = MRmaxIter;
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);
    open3d::pipelines::registration::GlobalOptimization(
        PoseGraph,open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
        criteria,
        option
    );
    if(verborse)
        std::cout << "Multiway Registration Finished." << std::endl;
}

void BackEndOptimizer::SaveMap(const std::string &filename)
{
    std::unique_lock<std::mutex> lock(FramePoseMutex);
    open3d::geometry::PointCloud Map;
    for(std::size_t i=0; i<FramePoses.size(); ++i)
    {
        Eigen::Isometry3d Pose = FramePoses[i];
        open3d::geometry::PointCloud laserCloud = LoadPCD(i);
        Map += laserCloud.Transform(Pose.matrix());
    }
    auto option = open3d::io::WritePointCloudOption(
        false,false,true
    );
    open3d::io::WritePointCloud(filename,*(Map.VoxelDownSample(voxel)),option);
    if(verborse)
        std::cout << "Map saved to " << filename << std::endl;
}