#pragma once
#include <ceres/ceres.h>
#include <vector>
#include <Eigen/Dense>
#include "GPR.hpp"
#include "g2o_tools.h"


inline bool allClose(const std::vector<double> &A, const std::vector<double> &B, const double &atol)
{
    assert(A.size() == B.size());
    for(int i = 0; i < A.size(); ++i)
    {
        if(abs(A[i] - B[i]) > atol)
            return false;
    }
    return true;
}

class IBAPlaneParams{
public:
    IBAPlaneParams(){};
public:
    double max_pixel_dist = 1.5;
    int num_best_convis = 3;
    int min_covis_weight = 100;
    int num_min_corr = 30;
    int kdtree2d_max_leaf_size = 10;
    int kdtree3d_max_leaf_size = 30;
    double norm_radius = 0.6;
    int norm_max_pts = 30;
    int max_iba_iter = 30;
    int inner_iba_iter = 10;
    double robust_kernel_delta = 2.98;
    double robust_kernel_3ddelta = 1.0;
    double sq_err_threshold = 225.;
    int PointCloudSkip = 1;
    bool PointCloudOnlyPositiveX = false;
    bool verborse = true;
};

class IBAGPRParams{
public:
    IBAGPRParams(){};
public:
    double max_pixel_dist = 1.5;
    int num_best_convis = 3;
    int min_covis_weight = 150;
    int num_min_corr = 30;
    int kdtree2d_max_leaf_size = 10;
    int kdtree3d_max_leaf_size = 30;
    double neigh_radius = 0.6;
    int neigh_max_pts = 30;
    int neigh_min_pts = 5;
    double robust_kernel_delta = 2.98;
    double init_sigma = 10.;
    double init_l = 10.;
    double sigma_noise = 1e-10;
    int PointCloudSkip = 1;
    bool PointCloudOnlyPositiveX = false;
    bool optimize_gpr = true;
    bool verborse = true;
};


class IBAGPR3dParams{
public:
    IBAGPR3dParams(){};
public:
    double max_pixel_dist = 1.5;
    double max_3d_dist = 1.0;
    double corr_3d_2d_threshold = 40.;
    double corr_3d_3d_threshold = 5.;
    double he_threshold = 0.05;
    int num_best_convis = 3;
    int min_covis_weight = 150;
    int num_min_corr = 30;
    int kdtree2d_max_leaf_size = 10;
    int kdtree3d_max_leaf_size = 30;
    double neigh_radius = 0.6;
    double norm_radius = 0.6;
    int norm_max_pts = 30;
    int norm_min_pts = 5;
    double min_diff_dist = 0.2;
    double norm_reg_threshold = 0.04;
    std::vector<double> err_weight = {1.0, 1.0};
    double pvalue = 3.0;
    double min_eval = 0.1;
    int neigh_max_pts = 30;
    int neigh_min_pts = 5;
    double robust_kernel_delta = 2.98;
    double robust_kernel_3ddelta = 1.0;
    double init_sigma = 10.;
    double init_l = 10.;
    double sigma_noise = 1e-10;
    int PointCloudSkip = 1;
    bool use_plane = true;
    bool PointCloudOnlyPositiveX = false;
    bool optimize_gpr = true;
    bool verborse = true;
};


class IBALocalParams{
public:
    IBALocalParams(){};
public:

    double max_pixel_dist = 1.5;
    double max_3d_dist = 1.0;
    double corr_3d_2d_threshold = 40.;
    double corr_3d_3d_threshold = 5.;
    double he_threshold = 0.05;
    int num_min_corr = 30;
    int num_best_convis = 3;
    int min_covis_weight = 150;
    int kdtree2d_max_leaf_size = 10;
    int kdtree3d_max_leaf_size = 30;
    double neigh_radius = 0.6;
    int neigh_max_pts = 30;
    int neigh_min_pts = 5;
    double min_diff_dist = 0.2;
    double norm_reg_threshold = 0.001;
    double pvalue = 3.0;
    double min_eval = 0.01;
    double robust_kernel_delta = 2.98;
    double robust_kernel_3ddelta = 1.0;
    double init_sigma = 10.;
    double init_l = 10.;
    double sigma_noise = 1e-10;
    int PointCloudSkip = 1;
    bool use_plane = true;
    bool PointCloudOnlyPositiveX = false;
    bool optimize_gpr = true;
    bool verborse = true;
    std::vector<double> err_weight = {1.0, 1.0};
};


struct IBA_PlaneFactor{
public:
    IBA_PlaneFactor(const int &_H, const int &_W, const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
     const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0):
     H(_H), W(_W), fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     p0(_p0), n0(_n0), NConv(_u1_list.size()), R_list(_R_list), t_list(_t_list)
     {};
    

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        T const* calib_sim3 = data[0];
        MatrixN<3, T> _Rcl;
        VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        VectorN<3, T> _p0 = p0.cast<T>();
        VectorN<3, T> _n0 = n0.cast<T>();
        // Manifold Transform
        VectorN<3, T> _p0c = _Rcl * _p0 + _tcl;
        VectorN<3, T> _n0c = _Rcl * _n0;
        T _Cxz = (_u0 - _cx)/_fx;
        T _Cyz = (_v0 - _cy)/_fy;
        T _Z0 = _n0c.dot(_p0c) / (_Cxz*_n0c(0) + _Cyz*_n0c(1) + _n0c(2));
        T _X0 = _Cxz * _Z0;
        T _Y0 = _Cyz * _Z0;
        VectorN<3, T> _P0(_X0, _Y0, _Z0);
        for(int i = 0; i < NConv; ++i)
        {
            MatrixN<3, T> _R = R_list[i].cast<T>();
            VectorN<3, T> _t = t_list[i].cast<T>();
            _t *= _s;
            T _u1(u1_list[i]), _v1(v1_list[i]);
            VectorN<3, T> _P1 = _R * _P0 + _t;  // Transform corresponding 3D point to Target camera coord
            T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;  // Reprojection Error
            error[2*i+1] = _v1_obs - _v1;
        }
        return true;
    }
        /**
     * @brief parameter blocks: extrinsic (7), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
     * The numResiduals will be automatically set according to the size of _u1_list
     * 
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 1st corresponding point x
     * @param _v0 1st corresponding point y
     * @param _u1_list list of 2nd corresponding point x
     * @param _v1_list list of 2nd corresponding point y
     * @param _R_list list of relative pose (Rotation)
     * @param _t_list list of relative pose (translation)
     * @param _p0 Manifold point (on the plane, LIDAR Coord System)
     * @param _n0 normal of _p0 (in the LIDAR Coord System)
     */
    static ceres::CostFunction *Create(const int &_H, const int &_W, const double &_fx, const double &_fy, const double &_cx, const double &_cy,
        const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
        const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0)
    {
        ceres::DynamicAutoDiffCostFunction<IBA_PlaneFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<IBA_PlaneFactor, 6>(
            new IBA_PlaneFactor(_H, _W, _fx, _fy, _cx, _cy, _u0, _v0, _u1_list, _v1_list, _R_list, _t_list ,_p0, _n0));
        cost_func->SetNumResiduals(2*_u1_list.size());
        cost_func->AddParameterBlock(7);  // only one block (extrinsic Sim3 log)
        return cost_func;
    }
private:
    const int H, W;
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> v1_list, u1_list;
    const Eigen::Vector3d p0;
    const Eigen::Vector3d n0;
    const int NConv;
    const std::vector<Eigen::Matrix3d> R_list;
    const std::vector<Eigen::Vector3d> t_list;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct IBA_Plane3dFactor{
public:
    IBA_Plane3dFactor(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
     const Eigen::Vector3d &_MapPoint, const Eigen::Matrix4d &_Tcw,
     const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     p0(_p0), n0(_n0), NConv(_u1_list.size()), R_list(_R_list), t_list(_t_list),
     MapPoint(_MapPoint), Tcw(_Tcw)
     {};
    

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        T const* calib_sim3 = data[0];
        MatrixN<3, T> _Rcl;
        VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        VectorN<3, T> _p0 = p0.cast<T>();
        VectorN<3, T> _n0 = n0.cast<T>();
        // Manifold Transform
        VectorN<3, T> _p0c = _Rcl * _p0 + _tcl;
        VectorN<3, T> _n0c = _Rcl * _n0;
        T _Cxz = (_u0 - _cx)/_fx;
        T _Cyz = (_v0 - _cy)/_fy;
        T _Z0 = _n0c.dot(_p0c) / (_Cxz*_n0c(0) + _Cyz*_n0c(1) + _n0c(2));
        T _X0 = _Cxz * _Z0;
        T _Y0 = _Cyz * _Z0;
        VectorN<3, T> _P0(_X0, _Y0, _Z0);
        for(int i = 0; i < NConv; ++i)
        {
            MatrixN<3, T> _R = R_list[i].cast<T>();
            VectorN<3, T> _t = t_list[i].cast<T>();
            _t *= _s;
            T _u1(u1_list[i]), _v1(v1_list[i]);
            VectorN<3, T> _P1 = _R * _P0 + _t;  // Transform corresponding 3D point to Target camera coord
            T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;  // Reprojection Error
            error[2*i+1] = _v1_obs - _v1;
        }
        VectorN<3, T> _MapPoint = MapPoint.cast<T>() * _s;
        MatrixN<4, T> _Tcw = Tcw.cast<T>();
        _MapPoint = _Tcw.topLeftCorner(3, 3) * _MapPoint + _Tcw.topRightCorner(3, 1) * _s;
        error[2*NConv + 1] = _MapPoint(0) - _P0(0);
        error[2*NConv + 2] = _MapPoint(1) - _P0(1);
        error[2*NConv + 3] = _MapPoint(2) - _P0(2);
        return true;
    }
        /**
     * @brief parameter blocks: extrinsic (7), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
     * The numResiduals will be automatically set according to the size of _u1_list
     * 
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 1st corresponding point x
     * @param _v0 1st corresponding point y
     * @param _u1_list list of 2nd corresponding point x
     * @param _v1_list list of 2nd corresponding point y
     * @param _R_list list of relative pose (Rotation)
     * @param _t_list list of relative pose (translation)
     * @param _MapPoint mapPoint camera world Pose
     * @param _Tcw Referece Frame pose (world to camera)
     * @param _p0 Manifold point (on the plane, LIDAR Coord System)
     * @param _n0 normal of _p0 (in the LIDAR Coord System)
     */
    static ceres::CostFunction *Create(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
        const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
        const Eigen::Vector3d &_MapPoint, const Eigen::Matrix4d &_Tcw,
        const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0)
    {
        ceres::DynamicAutoDiffCostFunction<IBA_Plane3dFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<IBA_Plane3dFactor, 6>(
            new IBA_Plane3dFactor(_fx, _fy, _cx, _cy, _u0, _v0, _u1_list, _v1_list, _R_list, _t_list ,_MapPoint, _Tcw, _p0, _n0));
        cost_func->SetNumResiduals(2*_u1_list.size() + 3); // 2n for IBA ,3 for 3d error
        cost_func->AddParameterBlock(7);  // only one block (extrinsic Sim3 log)
        return cost_func;
    }
private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> v1_list, u1_list;
    const Eigen::Vector3d p0;
    const Eigen::Vector3d n0;
    const int NConv;
    const std::vector<Eigen::Matrix3d> R_list;
    const std::vector<Eigen::Vector3d> t_list;
    const Eigen::Vector3d MapPoint;
    const Eigen::Matrix4d Tcw;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



struct IBA_SphereFactor{
public:
    IBA_SphereFactor(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
     const Eigen::Vector3d &_p0, const double &_r0):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     p0(_p0), r0(_r0), NConv(_u1_list.size()), R_list(_R_list), t_list(_t_list)
     {};
    

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        T const* calib_sim3 = data[0];
        MatrixN<3, T> _Rcl;
        VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        VectorN<3, T> _p0 = _Rcl * p0.cast<T>() + _tcl; // transform to camera coord
        T _r0(r0);
        T _k1 = (_u0 - _cx) / _fx;
        T _k2 = (_v0 - _cy) / _fy;
        T _a = (_k1 * _k1 + _k2 * _k2 + T(1));
        T _b = T(-2) * (_k1 * _p0(0) + _k2 * _p0(1) + _p0(2));
        T _c = _p0(0) * _p0(0) + _p0(1) * _p0(1) + _p0(2) * _p0(2) - _r0 * _r0;
        T _delta = _b * _b - T(4) * _a * _c;
        if(_delta < T(0))
        {
            return false;
        }
        T _Z0 = (-_b + sqrt(_delta)) / _a * T(0.5);
        if(_Z0 <= 0)
        {
            // for(int i = 0; i < NConv; ++i)
            // {
            //     error[2*i] = T(0);
            //     error[2*i+1] = T(0);
            // }
            // return true;
            return false;
        }
        T _X0 = _k1 * _Z0;
        T _Y0 = _k2 * _Z0;
        VectorN<3, T> _P0(_X0, _Y0, _Z0);
        for(int i = 0; i < NConv; ++i)
        {
            MatrixN<3, T> _R = R_list[i].cast<T>();
            VectorN<3, T> _t = t_list[i].cast<T>();
            _t *= _s;
            T _u1(u1_list[i]), _v1(v1_list[i]);
            VectorN<3, T> _P1 = _R * _P0 + _t;
            T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;
            error[2*i+1] = _v1_obs - _v1;
        }
        return true;
    }
        /**
     * @brief parameter blocks: extrinsic (7), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
     * The numResiduals will be automatically set according to the size of _u1_list
     * 
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 1st corresponding point x
     * @param _v0 1st corresponding point y
     * @param _u1_list list of 2nd corresponding point x
     * @param _v1_list list of 2nd corresponding point y
     * @param _R_list list of relative pose (Rotation)
     * @param _t_list list of relative pose (translation)
     * @param _p0 Manifold point (on the sphere, Lidar coord)
     * @param _r0 radius of sphere
     */
    static ceres::CostFunction *Create(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
        const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
        const Eigen::Vector3d &_p0, const double &_r0)
    {
        ceres::DynamicAutoDiffCostFunction<IBA_SphereFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<IBA_SphereFactor, 6>(
            new IBA_SphereFactor(_fx, _fy, _cx, _cy, _u0, _v0, _u1_list, _v1_list, _R_list, _t_list ,_p0, _r0));
        cost_func->SetNumResiduals(2*_u1_list.size());
        cost_func->AddParameterBlock(7);  // only one block (extrinsic Sim3 log)
        return cost_func;
    }
private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> v1_list, u1_list;
    const Eigen::Vector3d p0;
    const double r0;
    const int NConv;
    const std::vector<Eigen::Matrix3d> R_list;
    const std::vector<Eigen::Vector3d> t_list;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct IBA_GPRFactor{

public:
    IBA_GPRFactor(const double &_sigma, const double &_l, const double _sigma_noise,
     const std::vector<Eigen::Vector3d> &_neigh_pts, const Eigen::Matrix4d &init_SE3, const int &_H, const int &_W,
     const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list, 
     const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
     const bool _optimize=false, const bool _verborse=false):
     sigma(_sigma), l(_l), sigma_noise(_sigma_noise), neigh_pts(_neigh_pts),
     H(_H), W(_W), fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     NConv(u1_list.size()), R_list(_R_list), t_list(_t_list)
     {
        GPRParams gpr_params;
        gpr_params.sigma = _sigma;
        gpr_params.l = _l;
        gpr_params.verborse = _verborse;
        gpr_params.optimize = _optimize;
        std::vector<Eigen::Vector2d> train_x;  // transformed neighbour points
        Eigen::VectorXd train_y;
        train_x.resize(neigh_pts.size());
        train_y.resize(neigh_pts.size());
        double real_z = 0;
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            Eigen::Vector3d pt = init_SE3.topLeftCorner(3,3) * _neigh_pts[neigh_i] + init_SE3.topRightCorner(3,1);
            Eigen::Vector2d uv = {_fx * pt(0) / pt(2) + _cx, _fy * pt(1) / pt(2) + _cy};
            train_x[neigh_i] = uv;
            train_y[neigh_i] = pt(2);
            if(neigh_i == 0)
                real_z = pt(2);
        }
        GPR gpr(gpr_params);
        std::tie(sigma,l) = gpr.fit(train_x, train_y);
        if(_verborse)
        {
            Eigen::VectorXd test_x;
            test_x.resize(2);
            test_x << u0, v0;
            double pz = gpr.predict(test_x);
            std::printf("Real z:%lf, predict z:%lf\n",real_z, pz);
        }
     }

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        T const* calib_sim3 = data[0];
        MatrixN<3, T> _Rcl;
        VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        std::vector<VectorN<2, T>> _train_x;  // transformed neighbour points
        MatrixX<T> _train_y;
        VectorN<2, T> _test_x = {_u0, _v0};
        _train_x.resize(neigh_pts.size());
        _train_y.resize(neigh_pts.size(), 1);
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            VectorN<3 ,T> _tf_pt = _Rcl * neigh_pts[neigh_i].cast<T>() + _tcl;
            VectorN<2, T> _uv = {_fx * _tf_pt(0) / _tf_pt(2) + _cx, _fy * _tf_pt(1) / _tf_pt(2) + _cy}; /*X/Z, Y/Z*/
            _train_x[neigh_i] = _uv;
            _train_y(neigh_i) = _tf_pt(2);
        }
        TGPR gpr(sigma_noise, sigma, l, false);
        T _test_z = gpr.fit_predict<T>(_train_x, _train_y, _test_x);
        VectorN<3, T> _P0 = {_test_z * (_u0 - _cx) / _fx, _test_z * (_v0 - _cy) / _fy, _test_z};
        for(int i = 0; i < NConv; ++i)
        {
            MatrixN<3, T> _R = R_list[i].cast<T>();
            VectorN<3, T> _t = t_list[i].cast<T>() * _s;
            T _u1(u1_list[i]), _v1(v1_list[i]);
            VectorN<3, T> _P1 = _R * _P0 + _t;
            T _u1_obs = _fx * _P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy * _P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;
            error[2*i+1] = _v1_obs - _v1;
        }
        return true;
    }

    /**
     * @brief parameter blocks: extrinsic (7), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
     * The numResiduals will be automatically set according to the size of _u1_list
     * 
     * @param _sigma hyperparamter 1: Amplitude of RBF Kernel
     * @param _l hyperparamter 2: lenght scale of RBF Kernel
     * @param _sigma_noise variance of noise to ensure PSD (1e-10)
     * @param _neigh_pts neighbor points around target 3D point (in the LIDAR Coord System)
     * @param _init_SE3 initial Tcl for GPR hyperparamter adaptation
     * @param _H image height
     * @param _W image width
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 first correspondence point x
     * @param _v0 first correspondence point y
     * @param _u1_list second correspondence point x
     * @param _v1_list second correspondence point y
     * @param _R_list list of relative pose (Rotation)
     * @param _t_list list of relative pose (translation)
     * @param _optimize whether to optimize GPR kenerl hyperparameters
     * @param _verborse open it while debugging
     * @param _lb lower bound
     * @param _ub upper bound
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction *Create(const double &_sigma, const double &_l, const double _sigma_noise,
        const std::vector<Eigen::Vector3d> &_neigh_pts, const Eigen::Matrix4d &_init_SE3, const int &_H, const int &_W,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy,
        const double &_u0, const double &_v0, 
        const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
        const bool _optimize=false, const bool _verborse=false)
     {
        ceres::DynamicAutoDiffCostFunction<IBA_GPRFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<IBA_GPRFactor, 6>(
            new IBA_GPRFactor(_sigma, _l, _sigma_noise, _neigh_pts, _init_SE3, _H, _W, _fx, _fy, _cx, _cy,
            _u0, _v0, _u1_list, _v1_list, _R_list, _t_list, _optimize, _verborse));
        cost_func->SetNumResiduals(2*_u1_list.size());
        cost_func->AddParameterBlock(7);
        return cost_func;
     }

private:
    const double fx, fy, cx, cy, u0, v0;
    const int H, W;
    const std::vector<double> u1_list, v1_list;
    const std::vector<Eigen::Vector3d> neigh_pts;
    double sigma, l, sigma_noise; // We have NOT added hyperparamter adaptation during Edge Optimization YET
    const int NConv;
    const std::vector<Eigen::Matrix3d> R_list;
    const std::vector<Eigen::Vector3d> t_list;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Point2Point_Factor{
public:
    Point2Point_Factor(const Eigen::Vector3d &_MapPoint, const Eigen::Vector3d &_QueryPoint):
        MapPoint(_MapPoint), QueryPoint(_QueryPoint){}
    template <typename T>
    bool operator()(const T* data, T* error) const
    {
        T inv_se3_calib[6] = {-data[0], -data[1], -data[2], -data[3], -data[4], -data[5]};
        MatrixN<3, T>  _Rlc;
        VectorN<3, T>  _tlc;
        T _s = data[6];
        std::tie(_Rlc, _tlc) = SE3Exp<T>(inv_se3_calib);
        VectorN<3, T> _MapPoint = _Rlc * (MapPoint.cast<T>() * _s) + _tlc;
        VectorN<3, T> _QueryPoint = QueryPoint.cast<T>();
        error[0] = _MapPoint[0] - _QueryPoint[0];
        error[1] = _MapPoint[1] - _QueryPoint[1];
        error[2] = _MapPoint[2] - _QueryPoint[2];
        return true;
    }
    /**
     * @brief Construct Point2Point Align Error for Tcl optimization
     * 
     * @param _MapPoint Pose of MapPoint in camera coord
     * @param _QueryPoint Matched Lidar Point in lidar coord
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction *Create(const Eigen::Vector3d &_MapPoint, const Eigen::Vector3d &_QueryPoint){
        ceres::AutoDiffCostFunction<Point2Point_Factor, 3, 7> *cost_func = new ceres::AutoDiffCostFunction<Point2Point_Factor, 3, 7>(
            new Point2Point_Factor(_MapPoint, _QueryPoint)
        );
        return cost_func;
    }

private:
    const Eigen::Vector3d MapPoint;
    const Eigen::Vector3d QueryPoint;
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Point2Plane_Factor{
public:
    Point2Plane_Factor(const Eigen::Vector3d &_MapPoint, const Eigen::Vector3d &_QueryPoint, const Eigen::Vector3d &_normal):
        MapPoint(_MapPoint), QueryPoint(_QueryPoint), normal(_normal){}
    template <typename T>
    bool operator()(const T* data, T* error) const
    {
        T inv_se3_calib[6] = {-data[0], -data[1], -data[2], -data[3], -data[4], -data[5]};
        MatrixN<3, T> _Rcl, _Rlc;
        VectorN<3, T> _tcl, _tlc;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data);
        std::tie(_Rlc, _tlc) = SE3Exp<T>(inv_se3_calib);
        VectorN<3, T> _MapPoint = _Rlc * (MapPoint.cast<T>() * _s) + _tlc;
        VectorN<3, T> _QueryPoint = QueryPoint.cast<T>();
        VectorN<3, T> _normal = normal.cast<T>();
        error[0] = (_MapPoint - _QueryPoint).dot(_normal);
        return true;
    }

    /**
     * @brief Construct Point2Point Align Error for Tcl optimization
     * 
     * @param _MapPoint Pose of MapPoint in camera coord
     * @param _QueryPoint Matched Lidar Point in lidar coord
     * @param _normal normal of _QueryPoint
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction *Create(const Eigen::Vector3d &_MapPoint, const Eigen::Vector3d &_QueryPoint, const Eigen::Vector3d &_normal){
        ceres::AutoDiffCostFunction<Point2Plane_Factor, 1, 7> *cost_func = new ceres::AutoDiffCostFunction<Point2Plane_Factor, 1, 7>(
            new Point2Plane_Factor(_MapPoint, _QueryPoint, _normal)
        );
        return cost_func;
    }

private:
    const Eigen::Vector3d MapPoint;
    const Eigen::Vector3d QueryPoint;
    const Eigen::Vector3d normal;
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

struct IBA_GPR3dFactor{

public:
    IBA_GPR3dFactor(const double &_sigma, const double &_l, const double _sigma_noise,
     const std::vector<Eigen::Vector3d> &_neigh_pts, const Eigen::Matrix4d &init_SE3, const int &_H, const int &_W,
     const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list, 
     const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
     const Eigen::Vector3d &_MapPoint,
     const bool _optimize=false, const bool _verborse=false,
     const Eigen::Vector2d &_lb=(Eigen::Vector2d() << 1e-3, 1e-3).finished(),
     const Eigen::Vector2d &_ub=(Eigen::Vector2d() << 1e3, 1e3).finished()):
     sigma(_sigma), l(_l), sigma_noise(_sigma_noise), neigh_pts(_neigh_pts),
     H(_H), W(_W), fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     NConv(u1_list.size()), R_list(_R_list), t_list(_t_list), MapPoint(_MapPoint)
     {
        GPRParams gpr_params;
        gpr_params.sigma = _sigma;
        gpr_params.l = _l;
        gpr_params.verborse = _verborse;
        gpr_params.lb = _lb;
        gpr_params.ub = _ub;
        gpr_params.optimize = _optimize;
        std::vector<Eigen::Vector2d> train_x;  // transformed neighbour points
        Eigen::VectorXd train_y;
        train_x.resize(neigh_pts.size());
        train_y.resize(neigh_pts.size());
        double real_z = 0;
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            Eigen::Vector3d pt = init_SE3.topLeftCorner(3,3) * _neigh_pts[neigh_i] + init_SE3.topRightCorner(3,1);
            Eigen::Vector2d uv = {_fx * pt(0) / pt(2) + _cx, _fy * pt(1) / pt(2) + _cy};
            train_x[neigh_i] = uv;
            train_y[neigh_i] = pt(2);
            if(neigh_i == 0)
                real_z = pt(2);
        }
        GPR gpr(gpr_params);
        std::tie(sigma,l) = gpr.fit(train_x, train_y);
        if(_verborse)
        {
            Eigen::VectorXd test_x;
            test_x.resize(2);
            test_x << u0, v0;
            double pz = gpr.predict(test_x);
            std::printf("Real z:%lf, predict z:%lf\n",real_z, pz);
        }
     }

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        T const* calib_sim3 = data[0];
        MatrixN<3, T> _Rcl;
        VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        std::vector<VectorN<2, T>> _train_x;  // transformed neighbour points
        MatrixX<T> _train_y;
        VectorN<2, T> _test_x = {_u0, _v0};
        _train_x.resize(neigh_pts.size());
        _train_y.resize(neigh_pts.size(), 1);
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            VectorN<3 ,T> _tf_pt = _Rcl * neigh_pts[neigh_i].cast<T>() + _tcl;
            VectorN<2, T> _uv = {_fx * _tf_pt(0) / _tf_pt(2) + _cx, _fy * _tf_pt(1) / _tf_pt(2) + _cy}; /*X/Z, Y/Z*/
            _train_x[neigh_i] = _uv;
            _train_y(neigh_i) = _tf_pt(2);
        }
        TGPR gpr(sigma_noise, sigma, l, false);
        T _test_z = gpr.fit_predict<T>(_train_x, _train_y, _test_x);
        VectorN<3, T> _P0 = {_test_z * (_u0 - _cx) / _fx, _test_z * (_v0 - _cy) / _fy, _test_z};
        for(int i = 0; i < NConv; ++i)
        {
            MatrixN<3, T> _R = R_list[i].cast<T>();
            VectorN<3, T> _t = t_list[i].cast<T>() * _s;
            T _u1(u1_list[i]), _v1(v1_list[i]);
            VectorN<3, T> _P1 = _R * _P0 + _t;
            T _u1_obs = _fx * _P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy * _P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;
            error[2*i+1] = _v1_obs - _v1;
        }
        VectorN<3, T> _MapPoint = MapPoint.cast<T>() * _s;  // MapPoint with real size
        error[2*NConv] = _MapPoint(0) - _P0(0);
        error[2*NConv + 1] = _MapPoint(1) - _P0(1);
        error[2*NConv + 2] = _MapPoint(2) - _P0(2);
        return true;
    }

    /**
     * @brief parameter blocks: extrinsic (7), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
     * The numResiduals will be automatically set according to the size of _u1_list
     * 
     * @param _sigma hyperparamter 1: Amplitude of RBF Kernel
     * @param _l hyperparamter 2: lenght scale of RBF Kernel
     * @param _sigma_noise variance of noise to ensure PSD (1e-10)
     * @param _neigh_pts neighbor points around target 3D point (in the LIDAR Coord System)
     * @param _init_SE3 initial Tcl for GPR hyperparamter adaptation
     * @param _H image height
     * @param _W image width
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 first correspondence point x
     * @param _v0 first correspondence point y
     * @param _u1_list second correspondence point x
     * @param _v1_list second correspondence point y
     * @param _R_list list of relative pose (Rotation)
     * @param _t_list list of relative pose (translation)
     * @param _MapPoint mapPoint (Ref Camera Coord)
     * @param _Tcw Referece Frame pose (world to camera)
     * @param _optimize whether to optimize GPR kenerl hyperparameters
     * @param _verborse open it while debugging
     * @param _lb lower bound
     * @param _ub upper bound
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction *Create(const double &_sigma, const double &_l, const double _sigma_noise,
        const std::vector<Eigen::Vector3d> &_neigh_pts, const Eigen::Matrix4d &_init_SE3, const int &_H, const int &_W,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy,
        const double &_u0, const double &_v0, 
        const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
        const Eigen::Vector3d &_MapPoint,
        const bool _optimize=false, const bool _verborse=false,
        const Eigen::Vector2d &_lb=(Eigen::Vector2d() << 1e-3, 1e-3).finished(),
        const Eigen::Vector2d &_ub=(Eigen::Vector2d() << 1e3, 1e3).finished())
     {
        ceres::DynamicAutoDiffCostFunction<IBA_GPR3dFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<IBA_GPR3dFactor, 6>(
            new IBA_GPR3dFactor(_sigma, _l, _sigma_noise, _neigh_pts, _init_SE3, _H, _W, _fx, _fy, _cx, _cy,
            _u0, _v0, _u1_list, _v1_list, _R_list, _t_list, _MapPoint, _optimize, _verborse, _lb, _ub
        ));
        cost_func->SetNumResiduals(2*_u1_list.size() + 3);  // 2n for IBA ,3 for 3d error
        cost_func->AddParameterBlock(7);
        return cost_func;
     }

private:
    const double fx, fy, cx, cy, u0, v0;
    const int H, W;
    const std::vector<double> u1_list, v1_list;
    const std::vector<Eigen::Vector3d> neigh_pts;
    double sigma, l, sigma_noise; // We have NOT added hyperparamter adaptation during Edge Optimization YET
    const int NConv;
    const std::vector<Eigen::Matrix3d> R_list;
    const std::vector<Eigen::Vector3d> t_list;
    const Eigen::Vector3d MapPoint;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct Quadratic_AlignFactor{
public:
    Quadratic_AlignFactor(const Eigen::Vector3d &_MapPoint,
     const Eigen::Matrix3d &_base, const double &_r1, const double &_r2,
     const Eigen::Vector3d &_nn_pt):
     MapPoint(_MapPoint), base(_base), r1(_r1), r2(_r2), nn_pt(_nn_pt){}

    template <typename T>
    bool operator()(T const* calib_sim3, T* error) const
    {
        T _r1(r1), _r2(r2);
        T inv_se3_calib[6] = {-calib_sim3[0], -calib_sim3[1], -calib_sim3[2], -calib_sim3[3], -calib_sim3[4], -calib_sim3[5]};
        VectorN<3, T> _nn_pt = nn_pt.cast<T>();
        VectorN<3, T> _nn_normal = base.row(2).cast<T>();
        MatrixN<3, T> _Rcl, _Rlc;
        VectorN<3, T> _tcl, _tlc;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        VectorN<3, T> _MapPoint = MapPoint.cast<T>() * _s;  // MapPoint with real size
        std::tie(_Rlc, _tlc) = SE3Exp<T>(inv_se3_calib);
        _MapPoint = _Rlc * _MapPoint + _tlc; // Lidar Coord
        VectorN<3, T> _FrenetPoint = base.cast<T>() * _MapPoint;  // Frenet Frame
        T _d = abs((_FrenetPoint - _nn_pt).dot(_nn_normal));  // point to plane distance (approximate distance to normal footpoint)
        T _k1 = sqrt(_d / (_d + _r1)), _k2 = sqrt(_d / (_d + _r2));
        error[0] = _k1 * _FrenetPoint[0];
        error[1] = _k2 * _FrenetPoint[1];
        error[2] = _FrenetPoint[2];
        return true;
    }
    /**
     * @brief Qudratic Alignment Metric. Compute the Qudratic distance between the MapPoint and the PointCloud Local Surface
     * 
     * @param _MapPoint MapPoint in the Reference KeyFrame
     * @param _base base transformation from global to Frenet Frame
     * @param _r1 principle radius (abs)
     * @param _r2 principle radius (abs), whose direction is perpendicular to _r1
     * @param _nn_pt the nearest (SO3 normal) point of the surface to the query
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction *Create(const Eigen::Vector3d &_MapPoint,
        const Eigen::Matrix3d &_base, const double &_r1, const double &_r2,
        const Eigen::Vector3d &_nn_pt)
    {
        ceres::AutoDiffCostFunction<Quadratic_AlignFactor, 3, 7> *cost_func = new ceres::AutoDiffCostFunction<Quadratic_AlignFactor, 3, 7>(
            new Quadratic_AlignFactor(_MapPoint, _base, _r1, _r2, _nn_pt)
        );
        return cost_func;
    }


private:
    const Eigen::Vector3d MapPoint; // MapPoint in Reference Camera Coord
    const Eigen::Matrix3d base;  // base vectors e1, e2, e3 to transform point from Cartesian to Frenet Frame
    const double r1,r2;  // curvature radii cooresponding to two principle curvatures
    const Eigen::Vector3d nn_pt;  // nearest point
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};







struct UIBA_PlaneFactor{

public:
    UIBA_PlaneFactor(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     p0(_p0), n0(_n0), NConv(_u1_list.size())
     {};
    

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        T const* calib_sim3 = data[0];
        T const* refpose = data[1];
        MatrixN<3, T> _Rcl, _Rref;
        VectorN<3, T> _tcl, _tref;
        MatrixN<4, T> _InvRefPose;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        std::tie(_Rref, _tref) = SE3Exp<T>(refpose);  // data[1] stores the Lie Algebra of Camera Reference Pose
        _InvRefPose.setIdentity();
        _InvRefPose.topLeftCorner(3, 3) = _Rref.transpose();
        _InvRefPose.topRightCorner(3, 1) = -_Rref.transpose() * _tref;
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        VectorN<3, T> _p0 = p0.cast<T>();
        VectorN<3, T> _n0 = n0.cast<T>();    
        for(int i = 0; i < NConv; ++i)
        {
            MatrixN<3, T> _R;
            VectorN<3, T> _t;
            std::tie(_R, _t) = SE3Exp<T>(data[i+2]);
            MatrixN<4, T> _Pose;
            _Pose.setIdentity();
            _Pose.topLeftCorner(3, 3) = _R;
            _Pose.topRightCorner(3, 1) = _t;
            MatrixN<4, T> _relPose = _Pose * _InvRefPose;
            _relPose.topRightCorner(3, 1) *= _s; // noalias
            T _u1(u1_list[i]), _v1(v1_list[i]);
            // Manifold Transform
            VectorN<3, T> _p0c = _Rcl * _p0 + _tcl;
            VectorN<3, T> _n0c = _Rcl * _n0;
            T _Cxz = (_u0 - _cx)/_fx;
            T _Cyz = (_v0 - _cy)/_fy;
            T _Z0 = _n0c.dot(_p0c) / (_Cxz*_n0c(0) + _Cyz*_n0c(1) + _n0c(2));
            T _X0 = _Cxz * _Z0;
            T _Y0 = _Cyz * _Z0;
            VectorN<3, T> _P0(_X0, _Y0, _Z0);
            VectorN<3, T> _P1 = _relPose.topLeftCorner(3,3) * _P0 + _relPose.topRightCorner(3, 1);
            T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;
            error[2*i+1] = _v1_obs - _v1;
        }
        return true;
    }
        /**
     * @brief parameter blocks: extrinsic (7), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
     * The numResiduals will be automatically set according to the size of _u1_list
     * 
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 1st corresponding point x
     * @param _v0 1st corresponding point y
     * @param _u1_list list of 2nd corresponding point x
     * @param _v1_list list of 2nd corresponding point y
     * @param _p0 corresponding 3D point (in the LIDAR Coord System)
     * @param _n0 normal of _p0 (in the LIDAR Coord System)
     */
    static ceres::CostFunction *Create(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
        const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0)
    {
        ceres::DynamicAutoDiffCostFunction<UIBA_PlaneFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<UIBA_PlaneFactor, 6>(
            new UIBA_PlaneFactor(_fx, _fy, _cx, _cy, _u0, _v0, _u1_list, _v1_list, _p0, _n0));
        cost_func->SetNumResiduals(2*_u1_list.size());
        cost_func->AddParameterBlock(7);
        for(int i = 0; i < _u1_list.size() + 1; ++ i)
            cost_func->AddParameterBlock(6); // Add Pose (RefPose + Convisible Pose)
        return cost_func;
    }
private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> v1_list, u1_list;
    const Eigen::Vector3d p0;
    const Eigen::Vector3d n0;
    const int NConv;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};



struct UIBA_GPRFactor{

public:
    UIBA_GPRFactor(const double &_sigma, const double &_l, const double _sigma_noise,
     const std::vector<Eigen::Vector3d> &_neigh_pts,
     const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0,
     const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const bool _optimize=false, const bool _verborse=false,
     const Eigen::Vector2d &_lb=(Eigen::Vector2d() << 1e-3, 1e-3).finished(),
     const Eigen::Vector2d &_ub=(Eigen::Vector2d() << 1e3, 1e3).finished()):
     sigma(_sigma), l(_l), sigma_noise(_sigma_noise), neigh_pts(_neigh_pts),
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     NConv(u1_list.size())
     {
        if(!_optimize)
            return;
        GPRParams gpr_params;
        gpr_params.sigma = _sigma;
        gpr_params.l = _l;
        gpr_params.verborse = _verborse;
        gpr_params.lb = _lb;
        gpr_params.ub = _ub;
        gpr_params.optimize = _optimize;
        std::vector<Eigen::Vector2d> train_x;  // transformed neighbour points
        train_x.resize(neigh_pts.size());
        VectorX<> train_y;
        train_y.resize(neigh_pts.size());
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            Eigen::Vector3d pt = _neigh_pts[neigh_i];
            Eigen::Vector2d uv = {pt(0) / pt(2), pt(1) / pt(2)};
            uv(0) = _fx * uv(0) + _cx;
            uv(1) = _fy * uv(1) + _cy;
            train_x[neigh_i] = uv;
            train_y(neigh_i) = pt(2);
        }
        GPR gpr(gpr_params);
        gpr.fit(train_x, train_y);
        sigma = gpr.sigma;
        l = gpr.l;
        if(_verborse)
        {
            char msg[100];
            sprintf(msg, "optimized gpr: sigma: %0.4lf, l: %0.4lf\n", sigma, l);
            std::cout << msg;
        }
     }

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        T const* calib_sim3 = data[0];
        T const* refpose = data[1];
        MatrixN<3, T> _Rcl, _Rref;
        VectorN<3, T> _tcl, _tref;
        MatrixN<4, T> _InvRefPose;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(calib_sim3); // data[0] stores the Lie Algebra of Extrinsic Matrix
        std::tie(_Rref, _tref) = SE3Exp<T>(refpose);  // data[1] stores the Lie Algebra of Camera Reference Pose
        _InvRefPose.setIdentity();
        _InvRefPose.topLeftCorner(3, 3) = _Rref.transpose();
        _InvRefPose.topRightCorner(3, 1) = -_Rref.transpose() * _tref;
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        std::vector<VectorN<2, T>> _train_x;  // transformed neighbour points
        MatrixX<T> _train_y;
        VectorN<2, T> _test_x = {_u0, _v0};
        _train_x.resize(neigh_pts.size());
        _train_y.resize(neigh_pts.size(), 1);
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            VectorN<3 ,T> _tf_pt = _Rcl * neigh_pts[neigh_i].cast<T>() + _tcl;
            VectorN<2, T> _uv = {_tf_pt(0) / _tf_pt(2), _tf_pt(1) / _tf_pt(2)}; /*X/Z, Y/Z*/
            _uv(0) = _fx * _uv(0) + _cx;
            _uv(1) = _fy * _uv(1) + _cy;
            _train_x[neigh_i] = _uv;
            _train_y(neigh_i) = _tf_pt(2);
        }
        TGPR gpr(sigma_noise, sigma, l, false);
        T _test_z = gpr.fit_predict<T>(_train_x, _train_y, _test_x);
        VectorN<3, T> _P0;
        _P0(0) = _test_z * (_u0 - _cx) / _fx;
        _P0(0) = _test_z * (_v0 - _cy) / _fy;
        _P0(0) = _test_z;
        for(int i = 0; i < NConv; ++i)
        {
            MatrixN<3, T> _R;
            VectorN<3, T> _t;
            std::tie(_R, _t) = SE3Exp<T>(data[i+2]);
            MatrixN<4, T> _Pose;
            _Pose.setIdentity();
            _Pose.topLeftCorner(3, 3) = _R;
            _Pose.topRightCorner(3, 1) = _t;
            MatrixN<4, T> _relPose = _Pose * _InvRefPose;
            _relPose.topRightCorner(3, 1) *= _s;
            T _u1(u1_list[i]), _v1(v1_list[i]);
            VectorN<3, T> _P1 = _relPose.topLeftCorner(3, 3) * _P0 + _relPose.topRightCorner(3, 1);
            T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;
            error[2*i+1] = _v1_obs - _v1;
        }
        return true;
    }

    /**
     * @brief parameter blocks: extrinsic (7), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
     * The numResiduals will be automatically set according to the size of _u1_list
     * 
     * @param _sigma hyperparamter 1: Amplitude of RBF Kernel
     * @param _l hyperparamter 2: lenght scale of RBF Kernel
     * @param _sigma_noise variance of noise to ensure PSD (1e-10)
     * @param _neigh_pts neighbor points around target 3D point (in the LIDAR Coord System)
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 first correspondence point x
     * @param _v0 first correspondence point y
     * @param _u1_list second correspondence point x
     * @param _v1_list second correspondence point y
     * @param _optimize whether to optimize GPR kenerl hyperparameters
     * @param _verborse open it while debugging
     * @param _lb lower bound
     * @param _ub upper bound
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction *Create(const double &_sigma, const double &_l, const double _sigma_noise,
        const std::vector<Eigen::Vector3d> &_neigh_pts,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy,
        const double &_u0, const double &_v0, 
        const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const bool _optimize=false, const bool _verborse=false,
        const Eigen::Vector2d &_lb=(Eigen::Vector2d() << 1e-3, 1e-3).finished(),
        const Eigen::Vector2d &_ub=(Eigen::Vector2d() << 1e3, 1e3).finished())
     {
        ceres::DynamicAutoDiffCostFunction<UIBA_GPRFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<UIBA_GPRFactor, 6>(
            new UIBA_GPRFactor(_sigma, _l, _sigma_noise, _neigh_pts, _fx, _fy, _cx, _cy,
            _u0, _v0, _u1_list, _v1_list, _optimize, _verborse, _lb, _ub
        ));
        cost_func->SetNumResiduals(2*_u1_list.size());
        cost_func->AddParameterBlock(7);
        for(int i = 0; i < _u1_list.size() + 1; ++ i)
            cost_func->AddParameterBlock(6);  // Add Pose (RefPose + Convisible Pose)
        return cost_func;
     }

private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> u1_list, v1_list;
    const std::vector<Eigen::Vector3d> neigh_pts;
    double sigma, l, sigma_noise; // We have NOT added hyperparamter adaptation during Edge Optimization YET
    const int NConv;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

class BACostFunction : public ceres::SizedCostFunction<1, 6, 3>{
public:
    /**
     * @brief Construct a new BACostFunction object
     * 
     * @param _MapPoint WorldPose of the MapPoint
     * @param _u0 measurement x coordinate of the keypoint
     * @param _v0 measurementyx coordinate of the keypoint
     * @param _fx focal x length
     * @param _fy focal y length
     * @param _cx principle x length
     * @param _cy principle y length
     */
    BACostFunction(const double &_u0, const double &_v0,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy):
         u0(_u0), v0(_v0), fx(_fx), fy(_fy), cx(_cx), cy(_cy){}
    
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        std::tie(rotation, translation) = SE3Exp<double>(parameters[0]);
        Eigen::Vector3d MptWorld(parameters[1]);
        Eigen::Vector3d P = rotation * MptWorld + translation;
        double invz = 1.0 / P.z();
        double invz2 = invz * invz;
        double u1 = fx * P.x() * invz + cx;
        double v1 = fy * P.y() * invz + cy;
        residuals[0] = u0 - u1;  // measurement - observation
        residuals[1] = v0 - v1;  // measurement - observation
        if (jacobians != nullptr)
        {
            if(jacobians[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > Jpose(jacobians[0]);
                Jpose.row(0) << fx * P.x() * P.y() * invz2, -fx * P.x() * P.x() * invz2, fx * P.y() * invz, -fx * invz, 0, fx * P.x() * invz2;
                Jpose.row(1) << fy + fy * P.y() * P.y() * invz2, -fy * P.x() * P.y() * invz2, -fy * P.x() * invz, 0, -fy * invz, fy * P.y() * invz2;

            }
            if(jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jpoint(jacobians[1]);
                Jpoint.row(0) << -fx * invz,   0,  fx * P.x() * invz2,
                Jpoint.row(1) << 0,  -fy * invz, fy * P.y() * invz2; 
                Jpoint = Jpoint * rotation;
            }
        }
        return true;
    }
private:
    const double u0, v0;
    const double fx, fy, cx, cy;
};

struct BA_Factor{
public:
    BA_Factor(const double &_u0, const double &_v0,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy):
         u0(_u0), v0(_v0), fx(_fx), fy(_fy), cx(_cx), cy(_cy){}

    template <typename T>
    bool operator()(const T* cam_pose, const T* map_point, T* error) const
    {
        VectorN<3, T> _MapPoint(map_point);
        MatrixN<3, T> _Rcw;
        VectorN<3, T> _tcw;
        std::tie(_Rcw, _tcw) = SE3Exp<T>(cam_pose);
        _MapPoint = _Rcw * _MapPoint + _tcw;
        T _u0(u0), _v0(v0), _fx(fx), _fy(fy), _cx(cx), _cy(cy);
        error[0] = _fx * _MapPoint[0] / _MapPoint[2] + _cx - _u0;
        error[1] = _fy * _MapPoint[1] / _MapPoint[2] + _cy - _v0;
        return true;
    }
    /**
     * @brief Construct a new BA_Factor object 
     * input param blocks: cam_pose (6), map_worldpose (3), 
     * Joint Optimization for cam_pose and map_point
     * @param _u0 measurement x coordinate of the keypoint
     * @param _v0 measurementyx coordinate of the keypoint
     * @param _fx focal x length
     * @param _fy focal y length
     * @param _cx principle x length
     * @param _cy principle y length
     */
    static ceres::CostFunction *Create(const double &_u0, const double &_v0,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy)
    {
        ceres::AutoDiffCostFunction<BA_Factor, 2, 6, 3> *cost_func = new ceres::AutoDiffCostFunction<BA_Factor, 2, 6, 3>(
            new BA_Factor(_u0, _v0, _fx, _fy, _cx, _cy)
        );
        return cost_func;
    }

private:
    const double u0, v0;
    const double fx, fy, cx, cy;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct BAStructure_Factor{
public:
    BAStructure_Factor(const Eigen::Matrix4d &_Tcw, const double &_u0, const double &_v0,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy):
         Tcw(_Tcw), u0(_u0), v0(_v0), fx(_fx), fy(_fy), cx(_cx), cy(_cy){}
    BAStructure_Factor(const double *Tcw_log, const double &_u0, const double &_v0,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy):
         u0(_u0), v0(_v0), fx(_fx), fy(_fy), cx(_cx), cy(_cy)
         {
            Tcw = SE3Exp4x4<double>(Tcw_log);
         }

    template <typename T>
    bool operator()(const T* map_point, T* error) const
    {
        VectorN<3, T> _MapPoint(map_point);
        MatrixN<4, T> _Tcw = Tcw.cast<T>();
        _MapPoint = _Tcw.topLeftCorner(3, 3) * _MapPoint + _Tcw.topRightCorner(3, 1);
        T _u0(u0), _v0(v0), _fx(fx), _fy(fy), _cx(cx), _cy(cy);
        error[0] = _fx * _MapPoint[0] / _MapPoint[2] + _cx - _u0;
        error[1] = _fy * _MapPoint[1] / _MapPoint[2] + _cy - _v0;
        return true;
    }
    /**
     * @brief Construct a new BAStructure_Factor object 
     * input param blocks: cam_pose (6), map_worldpose (3), 
     * optimization for Mappoint only
     * @param _Tcw Current Camera Pose
     * @param _u0 measurement x coordinate of the keypoint
     * @param _v0 measurementyx coordinate of the keypoint
     * @param _fx focal x length
     * @param _fy focal y length
     * @param _cx principle x length
     * @param _cy principle y length
     */
    static ceres::CostFunction *Create(const Eigen::Matrix4d &_Tcw, const double &_u0, const double &_v0,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy)
    {
        ceres::AutoDiffCostFunction<BAStructure_Factor, 2, 3> *cost_func = new ceres::AutoDiffCostFunction<BAStructure_Factor, 2, 3>(
            new BAStructure_Factor(_Tcw, _u0, _v0, _fx, _fy, _cx, _cy)
        );
        return cost_func;
    }

    /**
     * @brief Construct a new BAStructure_Factor object 
     * input param blocks: cam_pose (6), map_worldpose (3), 
     * optimization for Mappoint only
     * @param Tcw_log se3 log of Current Camera Pose
     * @param _u0 measurement x coordinate of the keypoint
     * @param _v0 measurementyx coordinate of the keypoint
     * @param _fx focal x length
     * @param _fy focal y length
     * @param _cx principle x length
     * @param _cy principle y length
     */
    static ceres::CostFunction *Create(const double *Tcw_log, const double &_u0, const double &_v0,
        const double &_fx, const double &_fy, const double &_cx, const double &_cy)
    {
        ceres::AutoDiffCostFunction<BAStructure_Factor, 2, 3> *cost_func = new ceres::AutoDiffCostFunction<BAStructure_Factor, 2, 3>(
            new BAStructure_Factor(Tcw_log, _u0, _v0, _fx, _fy, _cx, _cy)
        );
        return cost_func;
    }

private:
    Eigen::Matrix4d Tcw;
    const double u0, v0;
    const double fx, fy, cx, cy;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct CrossPt_Factor{
public:
    CrossPt_Factor(const Eigen::Vector3d &_QueryPoint): QueryPoint(_QueryPoint){}
    template <typename T>
    bool operator()(const T* Tcl_log, const T* cam_pose, const T* map_pose, T* error) const
    {
        T inv_se3_calib[6] = {-Tcl_log[0], -Tcl_log[1], -Tcl_log[2], -Tcl_log[3], -Tcl_log[4], -Tcl_log[5]};
        MatrixN<3, T> _Rlc, _Rcw;
        VectorN<3, T> _tlc, _tcw;
        T _s = Tcl_log[6];
        VectorN<3, T> _MapPoint(map_pose);
        std::tie(_Rlc, _tlc) = SE3Exp<T>(inv_se3_calib);
        std::tie(_Rcw, _tcw) = SE3Exp<T>(cam_pose);
        _MapPoint = _Rcw * _MapPoint + _tcw;
        VectorN<3, T> _MapPointLidar = _Rlc * (_MapPoint * _s) + _tlc;
        VectorN<3, T> _QueryPoint = QueryPoint.cast<T>();
        error[0] = _MapPointLidar[0] - _QueryPoint[0];
        error[1] = _MapPointLidar[1] - _QueryPoint[1];
        error[2] = _MapPointLidar[2] - _QueryPoint[2];
        return true;
    }
    /**
     * @brief Construct a Cross Modality Point-to-Point Error. Joint-optimization of camera pose, MapPoint pose, extrinsic parameters and scale
     * 
     * @param _QueryPoint Matched Lidar Point in Lidar Coord
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction *Create(const Eigen::Vector3d &_QueryPoint){
        ceres::AutoDiffCostFunction<CrossPt_Factor, 3, 7, 6, 3> *cost_func = new ceres::AutoDiffCostFunction<CrossPt_Factor, 3, 7, 6, 3>(
            new CrossPt_Factor(_QueryPoint)
        );
        return cost_func;
    }

private:
    const Eigen::Vector3d QueryPoint;  // Matched Lidar Point
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CrossPL_Factor{
public:
    CrossPL_Factor(const Eigen::Vector3d &_QueryPoint, const Eigen::Vector3d &_normal):
     QueryPoint(_QueryPoint), normal(_normal){}
    template <typename T>
    bool operator()(const T* Tcl_log, const T* cam_pose, const T* map_pose, T* error) const
    {
        T inv_se3_calib[6] = {-Tcl_log[0], -Tcl_log[1], -Tcl_log[2], -Tcl_log[3], -Tcl_log[4], -Tcl_log[5]};
        MatrixN<3, T> _Rlc, _Rcw;
        VectorN<3, T> _tlc, _tcw;
        T _s = Tcl_log[6];
        VectorN<3, T> _MapPoint(map_pose);
        std::tie(_Rlc, _tlc) = SE3Exp<T>(inv_se3_calib);
        std::tie(_Rcw, _tcw) = SE3Exp<T>(cam_pose);
        _MapPoint = _Rcw * _MapPoint + _tcw;  // World Pose to Camera Pose
        VectorN<3, T> _MapPointLidar = _Rlc * (_MapPoint * _s) + _tlc;  // Camera Pose to Lidar Pose
        VectorN<3, T> _QueryPoint = QueryPoint.cast<T>();
        VectorN<3, T> _normal = normal.cast<T>();
        error[0] = (_QueryPoint - _MapPointLidar).dot(_normal);
        return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector3d &_QueryPoint, const Eigen::Vector3d &_normal){
        ceres::AutoDiffCostFunction<CrossPL_Factor, 1, 7, 6, 3> *cost_func = new ceres::AutoDiffCostFunction<CrossPL_Factor, 1, 7, 6, 3>(
            new CrossPL_Factor(_QueryPoint, _normal)
        );
        return cost_func;
    }

private:
    const Eigen::Vector3d QueryPoint;  // Matched Lidar Point
    const Eigen::Vector3d normal; // normal of QueryPoint
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};