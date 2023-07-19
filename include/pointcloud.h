#pragma once
#include <Eigen/Dense>
#include <tuple>
#include <boost/functional/hash.hpp>
#include <unordered_map>
#include "KDTreeVectorOfVectorsAdaptor.h"

// C++ 17 or later does not need to manually indicate allocator anymore
// typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
// typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
// typedef std::vector<Eigen::Index, Eigen::aligned_allocator<Eigen::Index>> VecIndex;

typedef std::vector<Eigen::Vector2d> VecVector2d;
typedef std::vector<Eigen::Vector3d> VecVector3d;
typedef std::uint32_t IndexType; // other types cause error, why?
typedef std::vector<IndexType> VecIndex;
typedef std::pair<IndexType, IndexType> CorrType;
typedef std::vector<CorrType> CorrSet;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector2d, double, 2, nanoflann::metric_L2_Simple, IndexType> KDTree2D;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector3d, double, 3, nanoflann::metric_L2_Simple, IndexType> KDTree3D;
typedef nanoflann::KDTreeVectorOfVectorsAdaptor<VecVector3d, double, 3, nanoflann::metric_SO3, IndexType> KDTreeSO3;

namespace std
{

template<typename... T>
struct hash<std::tuple<T...>>
{
    size_t operator()(tuple<T...> const& arg) const noexcept
    {
        return boost::hash_value(arg);
    }
};

}

VecVector3d SelectPoints(const VecVector3d &points, const VecIndex &indices)
{
    VecVector3d subpoints(indices.size());
    for(std::size_t i = 0; i < indices.size(); ++i)
        subpoints[i] = points[indices[i]];
    return subpoints;
}

VecIndex UniquePoints(const VecVector3d &points, const VecIndex &indices)
{
    std::unordered_map<std::tuple<double, double, double>, IndexType, std::hash<std::tuple<double, double, double>>> local_pt_map;
    local_pt_map.reserve(indices.size());
    for(auto const &idx:indices)
    {
        const auto pt = points[idx];
        local_pt_map[{pt[0], pt[1], pt[2]}] = idx;
    }
    VecIndex unique_indices;
    unique_indices.reserve(local_pt_map.size());
    for(auto const &[key, value]:local_pt_map)
        unique_indices.push_back(value);
    return unique_indices;
}

void RotatePointCloudInplace(VecVector3d &pointCloud, const Eigen::Matrix3d &rotation){
    for(auto &point:pointCloud)
        point = rotation * point;  // noalias
}

void RotatePointCloud(const VecVector3d &srcPointCloud, VecVector3d &tgtPointCloud, const Eigen::Matrix3d &rotation){
    tgtPointCloud.resize(srcPointCloud.size());
    for(std::size_t i = 0; i<srcPointCloud.size(); ++i)
        tgtPointCloud[i] = rotation * srcPointCloud[i];
}

void TransformPointCloudInplace(VecVector3d &pointCloud, const Eigen::Isometry3d &transformation){
    for(auto &point:pointCloud)
        point = transformation * point; // noalias
}

void TransformPointCloudInplace(VecVector3d &pointCloud, const Eigen::Matrix4d &transformation){
    for(auto &point:pointCloud)
        point = transformation.topLeftCorner(3, 3) * point + transformation.topRightCorner(3, 1);
}

void TransformPointCloud(const VecVector3d &srcPointCloud, VecVector3d &tgtPointCloud, const Eigen::Isometry3d &transformation){
    tgtPointCloud.resize(srcPointCloud.size());
    for(std::size_t i = 0; i<srcPointCloud.size(); ++i)
        tgtPointCloud[i] = transformation * srcPointCloud[i];
}

void TransformPointCloud(const VecVector3d &srcPointCloud, VecVector3d &tgtPointCloud, const Eigen::Matrix4d &transformation){
    tgtPointCloud.resize(srcPointCloud.size());
    for(std::size_t i = 0; i<srcPointCloud.size(); ++i)
        tgtPointCloud[i] = transformation.topLeftCorner(3, 3) * srcPointCloud[i] + transformation.topRightCorner(3, 1);
}

/**
 * @brief Compute Three angles and the area of a Triangle
 * 
 * @param a edge opposite angle A
 * @param b edge opposite angle B
 * @param c edge opposite angle C
 * @return std::tuple<double, double, double, double> respective opposite angles of a,b,c, Area
 */
std::tuple<double, double, double, double> ComputeTriangleInfo(const Eigen::Vector3d &a, const Eigen::Vector3d &b, const Eigen::Vector3d &c)
{
    const Eigen::Vector3d ab = a - b;
    const Eigen::Vector3d ac = a - c;
    const Eigen::Vector3d bc = b - c;
    double norm_ab = ab.norm();
    double norm_ac = ac.norm();
    double norm_bc = bc.norm();
    double cosA = ab.dot(ac) / norm_ab / norm_ac;  // ab \cdot ac / (|ab| \cdot |ac|)
    double cosB = (-ab).dot(bc) / norm_ab / norm_bc; // ba \cdot bc / (|ba| \cdot |bc|)
    double cosC = (ac).dot(bc) / norm_ac / norm_bc; // ca \cdot cb / (|ca| \cdot |cb|)
    double area = 0.5 * std::sqrt(1 - cosA * cosA) * norm_ab * norm_ac;
    return {std::acos(cosA), std::acos(cosB), std::acos(cosC), area};
}


/**
 * @brief Compute Convariance of the points indexed by indices (zero-mean included)
 * 
 * @tparam IdxType 
 * @param points the whole point cloud
 * @param indices query points to be computed covariance
 * @return Eigen::Matrix3d 
 */
template <typename IdxType = IndexType>
Eigen::Matrix3d ComputeCovariance(const VecVector3d &points,
                                  const std::vector<IdxType> &indices) {
    if (indices.empty()) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto &idx : indices) {
        const Eigen::Vector3d &point = points[idx];
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }
    cumulants /= (double)indices.size();
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);
    return covariance;
}


template <typename IdxType = IndexType>
Eigen::Matrix3d ComputeCovariance(const VecVector3d &points) {
    if (points.empty()) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto &point : points) {
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }
    cumulants /= (double)points.size();
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);
    return covariance;
}

// Copied from Open3d: open3d/geometry/EstimateNormals.cpp
Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d &A, double eval0) {
    Eigen::Vector3d row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3d row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3d row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3d r0xr1 = row0.cross(row1);
    Eigen::Vector3d r0xr2 = row0.cross(row2);
    Eigen::Vector3d r1xr2 = row1.cross(row2);
    double d0 = r0xr1.dot(r0xr1);
    double d1 = r0xr2.dot(r0xr2);
    double d2 = r1xr2.dot(r1xr2);

    double dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }

    if (imax == 0) {
        return r0xr1 / std::sqrt(d0);
    } else if (imax == 1) {
        return r0xr2 / std::sqrt(d1);
    } else {
        return r1xr2 / std::sqrt(d2);
    }
}

Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d &A,
                                    const Eigen::Vector3d &evec0,
                                    double eval1) {
    Eigen::Vector3d U, V;
    if (std::abs(evec0(0)) > std::abs(evec0(1))) {
        double inv_length =
                1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
        U << -evec0(2) * inv_length, 0, evec0(0) * inv_length;
    } else {
        double inv_length =
                1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
        U << 0, evec0(2) * inv_length, -evec0(1) * inv_length;
    }
    V = evec0.cross(U);

    Eigen::Vector3d AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                       A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                       A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

    Eigen::Vector3d AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};

    double m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    double m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    double m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

    double absM00 = std::abs(m00);
    double absM01 = std::abs(m01);
    double absM11 = std::abs(m11);
    double max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = std::max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / std::sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            return m01 * U - m00 * V;
        } else {
            return U;
        }
    } else {
        max_abs_comp = std::max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / std::sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            return m11 * U - m01 * V;
        } else {
            return U;
        }
    }
}

Eigen::Vector3d FastEigen3x3(const Eigen::Matrix3d &covariance) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    Eigen::Matrix3d A = covariance;
    double max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return Eigen::Vector3d::Zero();
    }
    A /= max_coeff;

    double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3d eval;
        Eigen::Vector3d evec0;
        Eigen::Vector3d evec1;
        Eigen::Vector3d evec2;
        
        double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        double b00 = A(0, 0) - q;
        double b11 = A(1, 1) - q;
        double b22 = A(2, 2) - q;

        double p =
                std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        double c00 = b11 * b22 - A(1, 2) * A(1, 2);
        double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        double half_det = det * 0.5;
        half_det = std::min(std::max(half_det, -1.0), 1.0);

        double angle = std::acos(half_det) / (double)3;
        double const two_thirds_pi = 2.09439510239319549;
        double beta2 = std::cos(angle) * 2;
        double beta0 = std::cos(angle + two_thirds_pi) * 2;
        double beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < eval(0) && eval(2) < eval(1)) {
                A *= max_coeff;
                return evec2;
            }
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec0 = evec1.cross(evec2);
            return evec0;
        } else {
            evec0 = ComputeEigenvector0(A, eval(0));
            if (eval(0) < eval(1) && eval(0) < eval(2)) {
                A *= max_coeff;
                return evec0;
            }
            evec1 = ComputeEigenvector1(A, evec0, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec2 = evec0.cross(evec1);
            return evec2;
        }
    } else {
        A *= max_coeff;
        if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) {
            return Eigen::Vector3d(1, 0, 0);
        } else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) {
            return Eigen::Vector3d(0, 1, 0);
        } else {
            return Eigen::Vector3d(0, 0, 1);
        }
    }
}


std::tuple<Eigen::Vector3d, Eigen::Vector3d> FastEigen3x3_EV(const Eigen::Matrix3d &covariance) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    Eigen::Matrix3d A = covariance;
    double max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    }
    A /= max_coeff;

    double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3d eval;
        Eigen::Vector3d evec0;
        Eigen::Vector3d evec1;
        Eigen::Vector3d evec2;
        
        double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        double b00 = A(0, 0) - q;
        double b11 = A(1, 1) - q;
        double b22 = A(2, 2) - q;

        double p =
                std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        double c00 = b11 * b22 - A(1, 2) * A(1, 2);
        double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        double half_det = det * 0.5;
        half_det = std::min(std::max(half_det, -1.0), 1.0);

        double angle = std::acos(half_det) / (double)3;
        double const two_thirds_pi = 2.09439510239319549;
        double beta2 = std::cos(angle) * 2;
        double beta0 = std::cos(angle + two_thirds_pi) * 2;
        double beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < eval(0) && eval(2) < eval(1)) {
                A *= max_coeff;
                return {evec2, eval};
            }
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return {evec1, eval};
            }
            evec0 = evec1.cross(evec2);
            return {evec0, eval};
        } else {
            evec0 = ComputeEigenvector0(A, eval(0));
            if (eval(0) < eval(1) && eval(0) < eval(2)) {
                A *= max_coeff;
                return {evec0, eval};
            }
            evec1 = ComputeEigenvector1(A, evec0, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return {evec1, eval};
            }
            evec2 = evec0.cross(evec1);
            return {evec2, eval};
        }
    } else {
        A *= max_coeff;
        if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) {
            return {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d::Zero()};
        } else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) {
            return {Eigen::Vector3d(0, 1, 0), Eigen::Vector3d::Zero()};
        } else {
            return {Eigen::Vector3d(0, 0, 1), Eigen::Vector3d::Zero()};
        }
    }
}

/**
 * @brief Compute normal of each point in points indexed by indices
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param querys indices of selected points
 * @param radius radius to compute point covariance
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @return std::tuple<VecVector3d, VecIndex>  valid normals, valid indices of querys
 */
std::tuple<VecVector3d, VecIndex> ComputeLocalNormal(const VecVector3d &points, const KDTree3D* kdtree, const VecIndex &querys,
    const double &radius, const int &max_pts, const int &min_pts=3, const double &pvalue = 3.0, const double &min_eval=1e-2)
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
        k = std::distance(sq_dist.begin(),
                      std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                       radius * radius)); // iterator difference between start and last valid index
        indices.resize(k);
        sq_dist.resize(k);
        if(k < min_pts)
            continue;
        Eigen::Matrix3d covariance = ComputeCovariance(points, indices);
        Eigen::Vector3d normal, eval;
        std::tie(normal, eval) = FastEigen3x3_EV(covariance);
        std::vector<double> eigenvalues{eval[0], eval[1], eval[2]};
        std::sort(eigenvalues.begin(), eigenvalues.end());
        if(!(eigenvalues[2] > pvalue * eigenvalues[1] && eigenvalues[2] > pvalue * eigenvalues[0] && eigenvalues[2] > min_eval))
            continue; // Valid Plane Verification
        normals.push_back(normal.normalized()); // self-to-self distance is minimum
        valid_indices.push_back(i);
    }
    return {normals, valid_indices};
}


/**
 * @brief Compute normal of each point in points indexed by indices
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param querys indices of selected points
 * @param radius radius to compute point covariance
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @return std::tuple<VecVector3d, std::vector<VecIndex>, VecIndex> valid normals, vector of valid neighbor point indices, valid indices of querys
 */
std::tuple<VecVector3d, std::vector<VecIndex>, VecIndex> ComputeLocalNormalAndNeighbor(const VecVector3d &points, const KDTree3D* kdtree, const VecIndex &querys,
    const double &radius, const int &max_pts, const int &min_pts=3)
{
    VecVector3d normals;
    VecIndex valid_indices;
    std::vector<VecIndex> neigh_indices;
    normals.reserve(querys.size());
    valid_indices.reserve(querys.size());
    neigh_indices.reserve(querys.size());
    for (std::size_t i = 0; i < querys.size(); ++i){
        const IndexType idx = querys[i];
        VecIndex indices(max_pts);
        std::vector<double> sq_dist(max_pts);
        nanoflann::KNNResultSet<double, IndexType> resultSet(max_pts);
        resultSet.init(indices.data(), sq_dist.data());
        kdtree->index->findNeighbors(resultSet, points[idx].data(), nanoflann::SearchParameters());
        std::size_t k = resultSet.size();
        k = std::distance(sq_dist.begin(),
                      std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                       radius * radius)); // iterator difference between start and last valid index
        indices.resize(k);
        if(k < min_pts)
            continue;
        neigh_indices.push_back(indices);
        Eigen::Matrix3d covariance = ComputeCovariance(points, indices);
        Eigen::Vector3d normal, eval;
        std::tie(normal, eval) = FastEigen3x3_EV(covariance);
        std::vector<double> eigenvalues{eval[0], eval[1], eval[2]};
        std::sort(eigenvalues.begin(), eigenvalues.end());
        if(!(eigenvalues[2] > 3 * eigenvalues[1] && eigenvalues[2] > 3 * eigenvalues[0] && eigenvalues[2] > 1e-2))
            continue; // Valid Plane Verification
        normals.push_back(normal.normalized()); // self-to-self distance is minimum
        valid_indices.push_back(i);
    }
    return {normals,neigh_indices, valid_indices};
}

/**
 * @brief Compute the normal of the single "query" point in "points"
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param query index of query point
 * @param radius maximum radius to compute point covariance
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @param pvalue the maximum eigenvalue must > "pvalue" * other_eigenvalue
 * @param min_eval the maximum eigenvalue must > min_eval
 * @return std::tuple<Eigen::Vector3d, bool> normal, isvalid
 */
std::tuple<Eigen::Vector3d, bool> ComputeLocalNormalSingle(const VecVector3d &points, const VecIndex &neigh_indices, const IndexType &query,
    const double &radius, const int &max_pts, const int &min_pts=3, const double pvalue=3.0, const double min_eval=0.01)
{
    Eigen::Matrix3d covariance = ComputeCovariance(points, neigh_indices);
    Eigen::Vector3d normal, eval;
    std::tie(normal, eval) = FastEigen3x3_EV(covariance);
    std::vector<double> eigenvalues{eval[0], eval[1], eval[2]};
    std::sort(eigenvalues.begin(), eigenvalues.end());
    normal.normalize();
    if(!(eigenvalues[2] > pvalue * eigenvalues[1] && eigenvalues[2] > pvalue * eigenvalues[0] && eigenvalues[2] > min_eval))
        return {normal, false}; // Valid Plane Verification
    return {normal, true};
}

/**
 * @brief Compute the normal of the single "query" point in "points"
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param query index of query point
 * @param radius maximum radius to compute point covariance
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @param pvalue the maximum eigenvalue must > "pvalue" * other_eigenvalue
 * @param min_eval the maximum eigenvalue must > min_eval
 * @return std::tuple<Eigen::Vector3d, bool> normal, isvalid
 */
std::tuple<Eigen::Vector3d, bool> ComputeLocalNormalSingle(const VecVector3d &points, const KDTree3D* kdtree, const IndexType &query,
    const double &radius, const int &max_pts, const int &min_pts=3, const double pvalue=3.0, const double min_eval=0.01)
{
    VecIndex indices(max_pts);
    std::vector<double> sq_dist(max_pts);
    nanoflann::KNNResultSet<double, IndexType> resultSet(max_pts);
    resultSet.init(indices.data(), sq_dist.data());
    kdtree->index->findNeighbors(resultSet, points[query].data(), nanoflann::SearchParameters());
    std::size_t k = resultSet.size();
    k = std::distance(sq_dist.begin(),
                    std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                    radius * radius)); // iterator difference between start and last valid index
    indices.resize(k);
    sq_dist.resize(k);
    if(k < min_pts)
        return {(Eigen::Vector3d() << 0,0,1).finished(), false};
    Eigen::Matrix3d covariance = ComputeCovariance(points, indices);
    Eigen::Vector3d normal, eval;
    std::tie(normal, eval) = FastEigen3x3_EV(covariance);
    normal.normalize();
    std::vector<double> eigenvalues{eval[0], eval[1], eval[2]};
    std::sort(eigenvalues.begin(), eigenvalues.end());
    if(!(eigenvalues[2] > pvalue * eigenvalues[1] && eigenvalues[2] > pvalue * eigenvalues[0] && eigenvalues[2] > min_eval))
        return {normal, false}; // Valid Plane Verification
    return {normal, true};
}


std::tuple<Eigen::Vector3d, bool> ComputeLocalNormalSingle(const VecVector3d &local_pts, const double pvalue=3.0, const double min_eval=0.01)
{
    Eigen::Matrix3d covariance = ComputeCovariance(local_pts);
    Eigen::Vector3d normal, eval;
    std::tie(normal, eval) = FastEigen3x3_EV(covariance);
    normal.normalize();
    std::vector<double> eigenvalues{eval[0], eval[1], eval[2]};
    std::sort(eigenvalues.begin(), eigenvalues.end());
    if(!(eigenvalues[2] > pvalue * eigenvalues[1] && eigenvalues[2] > pvalue * eigenvalues[0] && eigenvalues[2] > min_eval))
        return {normal, false}; // Valid Plane Verification
    return {normal, true};
}


/**
 * @brief Compute the local Normal of a single point
 * 
 * @param points pointcloud
 * @param neigh_indices indices of "points" to be computed
 * @param query_pt query_point
 * @param reg_threshold regression error 
 * @return std::tuple<Eigen::Vector3d, bool> 
 */
std::tuple<Eigen::Vector3d, bool> ComputeLocalNormalSingleThre(const VecVector3d &points, const VecIndex &neigh_indices, const Eigen::Vector3d &query_pt,
   const double &reg_threshold)
{
    Eigen::Matrix3d covariance = ComputeCovariance(points, neigh_indices);
    Eigen::Vector3d normal;
    std::tie(normal, std::ignore) = FastEigen3x3_EV(covariance);
    normal.normalize();
    double reg_err = 0;
    for(auto const &neigh_idx:neigh_indices)
        reg_err += std::abs((points[neigh_idx] - query_pt).dot(normal));
    reg_err /= neigh_indices.size() - 1;
    if(reg_err < reg_threshold)
        return {normal, true}; // Valid Plane Verification
    else
        return {normal, false};
}

std::tuple<Eigen::Vector3d, bool> ComputeLocalNormalSingleThre(const VecVector3d &local_pts, const Eigen::Vector3d &query_pt, const double &reg_threshold)
{
    Eigen::Matrix3d covariance = ComputeCovariance(local_pts);
    Eigen::Vector3d normal;
    std::tie(normal, std::ignore) = FastEigen3x3_EV(covariance);
    normal.normalize();
    double reg_err = 0;
    for(auto const &pt:local_pts)
        reg_err += std::abs((pt - query_pt).dot(normal));
    reg_err /= local_pts.size() - 1;
    if(reg_err < reg_threshold)
        return {normal, true}; // Valid Plane Verification
    else
        return {normal, false};
}


/**
 * @brief Compute the normal of the single "query" point in "points"
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param query index of query point
 * @param max_radius maximum radius to compute point covariance
 * @param min_radius minimum radius does the neighbor contain
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @param pvalue the maximum eigenvalue must > "pvalue" * other_eigenvalue
 * @param min_eval the maximum eigenvalue must > min_eval
 * @return std::tuple<Eigen::Vector3d, bool> normal, isvalid
 */
std::tuple<Eigen::Vector3d, bool> ComputeLocalNormalSingleThre(const VecVector3d &points, const KDTree3D* kdtree, const Eigen::Vector3d &query_pt,
    const double &max_radius, const double &min_radius, const int &max_pts, const int &min_pts, const double &reg_threshold)
{
    VecIndex indices(max_pts);
    std::vector<double> sq_dist(max_pts);
    nanoflann::KNNResultSet<double, IndexType> resultSet(max_pts);
    resultSet.init(indices.data(), sq_dist.data());
    kdtree->index->findNeighbors(resultSet, query_pt.data(), nanoflann::SearchParameters());
    std::size_t k = resultSet.size();
    k = std::distance(sq_dist.begin(),
                    std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                    max_radius * max_radius)); // iterator difference between start and last valid index
    indices.resize(k);
    sq_dist.resize(k);
    if(k < min_pts || sq_dist[k-1] < min_radius * min_radius)
        return {(Eigen::Vector3d() << 0,0,1).finished(), false};
    else
        return ComputeLocalNormalSingleThre(points, indices, query_pt, reg_threshold);
}




/**
 * @brief Compute neighbors of of each point in points indexed by indices
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param querys indices of selected points
 * @param max_radius maximum radius to compute point covariance
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @return std::tuple<std::vector<VecIndex>, VecIndex> Vec of neighbors' indices, valid index of querys
 */
std::tuple<std::vector<VecIndex>, VecIndex> ComputeLocalNeighbor(const VecVector3d &points, const KDTree3D* kdtree, const VecIndex &querys,
    const double &max_radius, const double &min_radius, const int &max_pts, const int &min_pts=3)
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
        k = std::distance(sq_dist.begin(),
                      std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                       max_radius * max_radius)); // iterator difference between start and last valid index
        indices.resize(k);
        sq_dist.resize(k);
        // VecIndex final_indices = UniquePoints(points, indices);
        if(k < min_pts || sq_dist[k-1] < min_radius * min_radius)
            continue;
        neighbors.push_back(indices); // self-to-self distance is minimum
        valid_indices.push_back(i);
    }
    return {neighbors, valid_indices};
}



/**
 * @brief Compute neighbors of the single "query" point in "points"
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param query_point query point
 * @param max_radius maximum radius to compute point covariance
 * @param min_radius minimum radius does the neighbor contain
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @return std::tuple<VecIndex, bool> Vector of neighbor indices, isvalid
 */
std::tuple<VecIndex, bool> ComputeLocalNeighborSingle(const VecVector3d &points, const KDTree3D* kdtree, const Eigen::Vector3d &query_point,
    const double &max_radius, const double &min_radius, const int &max_pts, const int &min_pts=3)
{
    VecIndex indices(max_pts);
    std::vector<double> sq_dist(max_pts);
    nanoflann::KNNResultSet<double, IndexType> resultSet(max_pts);
    resultSet.init(indices.data(), sq_dist.data());
    kdtree->index->findNeighbors(resultSet, query_point.data(), nanoflann::SearchParameters());
    int k = resultSet.size();
    k = std::distance(sq_dist.begin(),
                    std::lower_bound(sq_dist.begin(), sq_dist.begin() + k,
                                    max_radius * max_radius)); // iterator difference between start and last valid index
    indices.resize(k);
    sq_dist.resize(k);
    if(k < min_pts || sq_dist[k-1] < min_radius * min_radius)
        return {indices, false};
    return {indices, true};
}

/**
 * @brief Compute neighbors of the single "query" point in "points"
 * 
 * @param points the whole point cloud
 * @param kdtree KDTree built with point cloud (Lidar Coord)
 * @param query index of the query point
 * @param radius radius to compute point covariance
 * @param max_pts maximum points of the local manifold
 * @param min_pts minimum points the local manifold should at least contain
 * @return std::tuple<VecIndex, bool> Vector of neighbor indices, isvalid
 */
std::tuple<VecIndex, bool> ComputeLocalNeighborSingle(const VecVector3d &points, const KDTree3D* kdtree, const IndexType &query,
    const double &radius, const int &max_pts, const int &min_pts=3)
{
    return ComputeLocalNeighborSingle(points, kdtree, points[query], radius, max_pts, min_pts);
}