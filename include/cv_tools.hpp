#pragma once
#include <string>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

class CameraParams{
public:
    cv::Mat K;  // Intrinsic Parameters: 3x3
    cv::Mat DistCoef;  // Distortion Parameters
    float fps;
    int nRGB;
public:
    CameraParams(const std::string strSettingPath):K(cv::Mat::eye(3,3,CV_32F)), DistCoef(4,1,CV_32F){
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        if(!fSettings.isOpened()){
            throw std::runtime_error("Setting File "+strSettingPath+" Cannot open.");
            return;
        }
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        
        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        fps = fSettings["Camera.fps"];
        if(fps==0)
        fps=30;
        nRGB = fSettings["Camera.RGB"];
    }
    void UndistortPoints(cv::InputArray InputPoints, cv::OutputArray OutputPoints){
        cv::undistortPoints(InputPoints, OutputPoints, K, DistCoef);
    }
    void ProjectPoints(cv::InputArray InputPoints, cv::OutputArray OutputPoints, bool Distortion){
        cv::Mat rvec(1,3,CV_32F,0.0);
        cv::Mat tvec(1,3,CV_32F,0.0);
        if(!Distortion){
            cv::projectPoints(InputPoints,rvec,tvec, K, cv::noArray(), OutputPoints);
        }
        else
            cv::projectPoints(InputPoints,rvec,tvec, K, DistCoef, OutputPoints);
    }
    void ProjectPoints(cv::InputArray InputPoints, cv::OutputArray OutputPoints, cv::InputArray rvec, cv::InputArray tvec, bool Distortion){
        if(!Distortion){
            cv::projectPoints(InputPoints, rvec, tvec, K, cv::noArray(), OutputPoints);
        }
        else
            cv::projectPoints(InputPoints, rvec, tvec, K, DistCoef, OutputPoints);
    }
};

cv::Mat SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<
        0, -v.at<float>(2), v.at<float>(1),
        v.at<float>(2),   0,    -v.at<float>(0),
        -v.at<float>(1),  v.at<float>(0),  0);
}