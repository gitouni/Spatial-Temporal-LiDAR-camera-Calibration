#pragma once
#include <ceres/ceres.h>
#include <vector>
#include <Eigen/Dense>
#include "GPR.hpp"
#include "g2o_tools.h"

struct UIBA_PlaneFactor{

public:
    UIBA_PlaneFactor(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     p0(_p0), n0(_n0), Npose(u1_list.size()){};
    

    template <typename T>
    bool operator()(T const* const* data, T* error) const{
        MatrixN<3, T> _Rcl, _Rref;
        VectorN<3, T> _tcl, _tref;
        MatrixN<4, T> _InvRefPose;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data[0]); // data[0] stores the Lie Algebra of Extrinsic Matrix
        std::tie(_Rref, _tref) = SE3Exp<T>(data[1]);  // data[1] stores the Lie Algebra of Camera Reference Pose
        _InvRefPose.setIdentity();
        _InvRefPose.topLeftCorner(3, 3) = _Rref.transpose();
        _InvRefPose.topRightCorner(3, 1) = -_Rref.transpose() * _tref;
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        VectorN<3, T> _p0 = p0.cast<T>();
        VectorN<3, T> _n0 = n0.cast<T>();
        const int N = u1_list.size();
        
        for(int i = 0; i < Npose; ++i)
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
     * @brief Input data List: extrinsic (6), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
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
        return cost_func;
    }
private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> v1_list, u1_list;
    const Eigen::Vector3d p0;
    const Eigen::Vector3d n0;
    const int Npose;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};



struct UIBA_GPRFactor{
    
public:
    UIBA_GPRFactor(const double &_sigma, const double &_l, const double _sigma_noise,
     const std::vector<Eigen::Vector3d> &_neigh_pts,
     const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const bool _optimize=false, const bool _verborse=false,
     const Eigen::Vector2d &_lb=(Eigen::Vector2d() << 1e-3, 1e-3).finished(),
     const Eigen::Vector2d &_ub=(Eigen::Vector2d() << 1e3, 1e3).finished()):
     sigma(_sigma), l(_l), sigma_noise(_sigma_noise), neigh_pts(_neigh_pts),
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     Npose(u1_list.size())
     {
        assert(Npose > 0);
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
        Eigen::MatrixXd train_y;
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            Eigen::Vector3d pt = _neigh_pts[neigh_i];
            Eigen::Vector2d uv = {pt(0) / pt(2), pt(1) / pt(2)};
            uv(0) = _fx * uv(0) + _cx;
            uv(1) = _fy * uv(1) + _cy;
            train_x.push_back(uv);
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
        MatrixN<3, T> _Rcl, _Rref;
        VectorN<3, T> _tcl, _tref;
        MatrixN<4, T> _InvRefPose;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data[0]); // data[0] stores the Lie Algebra of Extrinsic Matrix
        std::tie(_Rref, _tref) = SE3Exp<T>(data[1]);  // data[1] stores the Lie Algebra of Camera Reference Pose
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
        for(int i = 0; i < Npose; ++i)
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
     * @brief Input data List: extrinsic (6), refPose (6), Pose1 (6), .... ,PoseK(6) (must corresponding to _u1_list).
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
        const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
        const bool _optimize=false, const bool _verborse=false,
        const Eigen::Vector2d &_lb=(Eigen::Vector2d() << 1e-3, 1e-3).finished(),
        const Eigen::Vector2d &_ub=(Eigen::Vector2d() << 1e3, 1e3).finished())
     {
        ceres::DynamicAutoDiffCostFunction<UIBA_GPRFactor, 6> *cost_func = new ceres::DynamicAutoDiffCostFunction<UIBA_GPRFactor, 6>(
            new UIBA_GPRFactor(_sigma, _l, _sigma_noise, _neigh_pts, _fx, _fy, _cx, _cy,
            _u0, _v0, _u1_list, _v1_list, _optimize, _verborse, _lb, _ub
        ));
        cost_func->SetNumResiduals(2*_u1_list.size());
        return cost_func;
     }

private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> u1_list, v1_list;
    const std::vector<Eigen::Vector3d> neigh_pts;
    double sigma, l, sigma_noise; // We have NOT added hyperparamter adaptation during Edge Optimization YET
    const int Npose;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};
