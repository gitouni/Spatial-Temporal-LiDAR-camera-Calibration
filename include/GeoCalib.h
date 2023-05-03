#include <nanoflann.hpp>
#include <Eigen/Dense>
#include "orb_slam/include/KeyFrame.h"
#include <unordered_map>
#include "cv_tools.hpp"

void TransformPointCloudInplace(std::vector<Eigen::Vector3d> &pointCloud, const Eigen::Isometry3d &transformation){
    for(std::uint32_t i = 0; i< (std::uint32_t) pointCloud.size(); ++i)
        pointCloud[i] = transformation * pointCloud[i];
}

void TransformationPointCloud(const std::vector<Eigen::Vector3d> &srcPointCloud, std::vector<Eigen::Vector3d> &tgtPointCloud, const Eigen::Isometry3d &transformation){
    tgtPointCloud.resize(srcPointCloud.size());
    for(std::uint32_t i = 0; i<srcPointCloud.size(); ++i)
        tgtPointCloud[i] = transformation * srcPointCloud[i];
}

void computeCorrespondence(const std::vector<Eigen::Vector3d> *srcPointCloud, const std::vector<Eigen::Vector3d> *tgtPointCloud,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> &corrSet, const double maxDistance = 0.05){
    
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, std::vector<Eigen::Vector3d>>, std::vector<Eigen::Vector3d>, 3, std::uint32_t> KDTreeType;
    
    std::unique_ptr<KDTreeType> kdtree(new KDTreeType(3, *tgtPointCloud, {15}));  // max_leaf_size = 15
    for(std::uint32_t i = 0; i < srcPointCloud->size(); ++i){
        std::uint32_t num_res = 1;
        std::vector<std::uint32_t> query_index(num_res);
        std::vector<double> sq_dist(num_res, 1000);
        num_res = kdtree->knnSearch(srcPointCloud->at(i).data(), num_res, query_index.data(), sq_dist.data());
        if(num_res > 0 && sq_dist[0] <= maxDistance)
            corrSet.push_back(std::make_pair(i, query_index[0]));
    }
}

void computeCorrespondenceList(const std::vector<std::vector<Eigen::Vector3d>* > &srcPointCloudList, const std::vector<Eigen::Vector3d> *tgtPointCloud,
    std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>* > &corrSetList, const double maxDistance = 0.05){
    
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, std::vector<Eigen::Vector3d>>, std::vector<Eigen::Vector3d>, 3, std::uint32_t> KDTreeType;
    
    std::unique_ptr<KDTreeType> kdtree(new KDTreeType(3, *tgtPointCloud, {15}));  // max_leaf_size = 15
    corrSetList.reserve(srcPointCloudList.size());
    for(std::uint32_t cloud_i = 0; cloud_i < srcPointCloudList.size(); ++ cloud_i){
        std::vector<std::pair<std::uint32_t, std::uint32_t>>* corrSet(new std::vector<std::pair<std::uint32_t, std::uint32_t>>);
        for(std::uint32_t i = 0; i < srcPointCloudList[cloud_i]->size(); ++i){
            std::uint32_t num_res = 1;
            std::vector<std::uint32_t> query_index(num_res);
            std::vector<double> sq_dist(num_res, 1000);
            num_res = kdtree->knnSearch(srcPointCloudList[cloud_i]->at(i).data(), num_res, query_index.data(), sq_dist.data());
            if(num_res > 0 && sq_dist[0] <= maxDistance)
                corrSet->push_back(std::make_pair(i, query_index[0]));
        }
        corrSetList.push_back(corrSet);
    }
}



cv::Mat ComputeF12(ORB_SLAM2::KeyFrame *&pKF1, ORB_SLAM2::KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void computeInitialGeoError(const std::vector<std::vector<Eigen::Vector3d>* > pointCloudList, const std::vector<ORB_SLAM2::KeyFrame* > KFList,
    const std::vector<Eigen::Isometry3d> &poseList, const double maxDistance = 0.05){
    std::vector<std::vector<Eigen::Vector3d>* > pointCloudTransformedList;
    std::cout << "Transforming PointCloud to World Coordinate System" << std::endl;
    for(std::uint32_t i = 0; i < pointCloudList.size(); ++i){
        std::vector<Eigen::Vector3d>* pointCloudTransformed(new std::vector<Eigen::Vector3d>);
        TransformationPointCloud(*pointCloudList[i], *pointCloudTransformed, std::ref(poseList[i]));
        pointCloudTransformedList.push_back(pointCloudTransformed);
    }
    std::cout << "Build Correspondences between Transformed PointClouds based on ORB_SLAM Edge Connection" << std::endl;
    // Map mnId to Index in KFList
    std::unordered_map<std::uint32_t, std::uint32_t> KFidMap;
    for(std::uint32_t KFi = 0; KFi < KFList.size(); ++ KFi)
        KFidMap.insert(std::make_pair(KFList[KFi]->mnId, KFi));
    // Construct Lists of Correspondence Set for Each PointCloud
    for(std::uint32_t KFi = 0; KFi < KFList.size(); ++ KFi){
        const std::vector<ORB_SLAM2::KeyFrame*> vpNeighKFs = KFList[KFi]->GetBestCovisibilityKeyFrames(20);
        std::vector<std::vector<Eigen::Vector3d>* > subPointCloudTransformedList;
        // Add Corresponding PointClouds to a List and Compute Correspondences One to One;
        for(ORB_SLAM2::KeyFrame* KF: vpNeighKFs){
            auto it = KFidMap.find(KF->mnId);
            assert(it!=KFidMap.end());  // Should Not be Triggered
            subPointCloudTransformedList.push_back(pointCloudTransformedList[it->second]);
        }
        std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>* > ptCorrSetList;
        computeCorrespondenceList(subPointCloudTransformedList, pointCloudTransformedList[KFi], std::ref(ptCorrSetList), maxDistance);
    }
        

}