#pragma once
#include "pointcloud.h"
#include <opencv2/core/core.hpp>
#include <unordered_map>
#include <Eigen/Sparse>


/**
 * @brief PaintPointClor through projection
 * 
 * @param proj_points (u,v) of project points
 * @param img cv::uint_8 gray-scale image
 * @param weightsum_threshold if the distance to the nearest round pixel is lower than this value, do not use weighted inv-distance to compute
 * @param max_intensity 255 for UINT_8 Format Image
 * @return std::vector<double> 
 */
std::vector<double> PaintPointCloud(const VecVector2d &proj_points, 
    const cv::Mat &img, const int weightsum_threshold=0.1, const double max_intensity=255.)
{
    std::unordered_map<int, Eigen::Vector3d> res;
    std::vector<double> proj_intensity;
    const int H = img.rows, W = img.cols;
    for(int proj_i = 0; proj_i < static_cast<int>(proj_points.size()); ++proj_i)
    {
        double u = proj_points[proj_i].x();
        double v = proj_points[proj_i].y();
        int intu = std::round(u);
        int intv = std::round(v);
        double dist = sqrt((u-intu)*(u-intu) + (v-intv)*(v-intv));
        if(dist < weightsum_threshold)
        {
            proj_intensity[proj_i] = static_cast<double>(img.at<cv::uint8_t>(intv, intu)) / max_intensity;
        }else
        {
            int minX = std::max(0, intu-1);
            int maxX = std::min(W, intu+1);
            int minY = std::max(0, intv-1);
            int maxY = std::min(H, intv+1);
            double invd_topleft = 1.0 / sqrt((u-minX)*(u-minX)+(v-minY)*(v-minY));
            double invd_topright = 1.0 / sqrt((u-maxX)*(u-maxX)+(v-minY)*(v-minY));
            double invd_bottomleft = 1.0 / sqrt((u-minX)*(u-minX)+(v-maxY)*(v-maxY));
            double invd_bottomright = 1.0 / sqrt((u-maxX)*(u-maxX)+(v-maxY)*(v-maxY));
            double topleft = static_cast<double>(img.at<cv::uint8_t>(minX,minY));
            double topright = static_cast<double>(img.at<cv::uint8_t>(maxX,minY));
            double bottomleft = static_cast<double>(img.at<cv::uint8_t>(minX,maxY));
            double bottomright = static_cast<double>(img.at<cv::uint8_t>(maxX,maxY));
            proj_intensity[proj_i] = 
                ((invd_topleft * topleft) + (invd_topright * topright) + (invd_bottomleft * bottomleft) + (invd_bottomright * bottomright)) / 
                (invd_topleft + invd_topright + invd_bottomleft + invd_bottomright) / max_intensity;
        }
    }
    return proj_intensity;
}


/**
 * @brief Build Color Gradient for project points
 * 
 * @param points Raw Lidar Points
 * @param proj_points 
 * @param proj_indices 
 * @param img 
 * @param weightsum_threshold 
 * @param max_intensity 
 * @return std::unordered_map<int, Eigen::Vector3d> 
 */
std::unordered_map<int, Eigen::Vector3d> buildColorGradient(const VecVector3d &points, const KDTree3D *kdtree,
    const VecVector2d &proj_points, const VecIndex &proj_indices,
    const cv::Mat &img, const int weightsum_threshold=0.1, const double max_intensity=255.,
    double normal_radius=0.6, int normal_max_nn=30, int normal_min_nn=5)
{
    assert(proj_points.size() == proj_indices.size());
    std::vector<double> instensity = PaintPointCloud(proj_points, img, weightsum_threshold, max_intensity);
    VecIndex valid_indices;
    VecVector3d normals;
    std::vector<VecIndex> neigh_indices;
    std::tie(normals, neigh_indices, valid_indices) = ComputeLocalNormalAndNeighbor(points, kdtree, proj_indices, normal_radius, normal_max_nn, normal_min_nn);
}