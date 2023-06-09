#pragma once
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "g2o/core/auto_differentiation.h"
#include "g2o_tools.h"
#include <GPR.hpp>

class IBATestEdge: public g2o::BaseUnaryEdge<2, g2o::Vector2, VertexSim3>
{
public:
    /**
     * @brief Construct a new IBAPlaneEdge Object (Auto Derivate)
     * 
     * @param _fx intrinsic focal length x
     * @param _fy intrinsic focal length y
     * @param _cx intrinsic principal point x
     * @param _cy intrinsic principal point y
     * @param _u0 first correspondence point x
     * @param _v0 first correspondence point y
     * @param _u1 second correspondence point x
     * @param _v1 second correspondence point y
     * @param _p0 corresponding 3D point (in the LIDAR Coord System)
     * @param _R camera relative Rotation (2nd node to 1st node)
     * @param _t camera relative Translation (2nd node to 1st node)
     */
    IBATestEdge(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const double &_u1, const double &_v1,
     const Eigen::Vector3d &_p0, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_t):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1(_u1), v1(_v1),
     p0(_p0), R(_R), t(_t){};
    

    template <typename T>
    bool operator()(const T* data, T* error) const{
        g2o::MatrixN<3, T> _Rcl;
        g2o::VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data); // Template Sim3 Map
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0), _u1(u1), _v1(v1);
        g2o::VectorN<3, T> _p0 = p0.cast<T>();
        g2o::MatrixN<3, T> _R = R.cast<T>();
        g2o::VectorN<3 ,T> _t = t.cast<T>() * _s;
        g2o::VectorN<3, T> _p0c = _Rcl * _p0 + _tcl;
        g2o::VectorN<3, T> _p1c = _R * _p0c + _t;
        // Project Observation to Image
        T _u1_obs = _fx*_p1c(0)/_p1c(2) + _cx;
        T _v1_obs = _fy*_p1c(1)/_p1c(2) + _cy;
        error[0] = _u1_obs - _u1;
        error[1] = _v1_obs - _v1;
        return true;
    }
    
    virtual bool read(std::istream &in) override {return false;}
    virtual bool write(std::ostream &out) const override {return false;}
    
private:
    const double fx, fy, cx, cy, u0, v0, u1, v1;
    const Eigen::Vector3d p0;
    const Eigen::Matrix3d R;
    const Eigen::Vector3d t;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    G2O_MAKE_AUTO_AD_FUNCTIONS  // Define ComputeError and linearizeOplus from operator()
};


class IBAPlaneEdge: public g2o::BaseUnaryEdge<20, g2o::VectorN<20>, VertexSim3>
{
public:
    /**
     * @brief Construct a new IBAPlaneEdge Object (Auto Derivate)
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
     * @param _R_list list of camera relative Rotation (2nd node to 1st node)
     * @param _t_list list of camera relative Translation (2nd node to 1st node)
     */
    IBAPlaneEdge(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0,
     const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     p0(_p0), n0(_n0), R_list(_R_list), t_list(_t_list)
     {};
    

    template <typename T>
    bool operator()(const T* data, T* error) const{
        g2o::MatrixN<3, T> _Rcl;
        g2o::VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data); // Template Sim3 Map
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        g2o::VectorN<3, T> _p0 = p0.cast<T>();
        g2o::VectorN<3, T> _n0 = n0.cast<T>();
        const int N = u1_list.size();
        
        for(int i = 0; i < N; ++i){
            g2o::MatrixN<3, T> _R = R_list[i].cast<T>();
            g2o::VectorN<3 ,T> _t = t_list[i].cast<T>() * _s;
            T _u1(u1_list[i]), _v1(v1_list[i]);
            // Manifold Transform
            g2o::VectorN<3, T> _p0c = _Rcl * _p0 + _tcl;
            g2o::VectorN<3, T> _n0c = _Rcl * _n0;
        
            T _Cxz = (_u0 - _cx)/_fx;
            T _Cyz = (_v0 - _cy)/_fy;
            T _Z0 = _n0c.dot(_p0c) / (_Cxz*_n0c(0) + _Cyz*_n0c(1) + _n0c(2));
            T _X0 = _Cxz * _Z0;
            T _Y0 = _Cyz * _Z0;
            g2o::VectorN<3, T> _P0(_X0, _Y0, _Z0);
            g2o::VectorN<3, T> _P1 = _R * _P0 + _t;
            T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;
            error[2*i+1] = _v1_obs - _v1;
        }
        for(int i = N; i < 10; ++i)
        {
            error[2*i] = T(0);
            error[2*i+1] = T(0);
        }
        
        return true;
    }
    
    virtual bool read(std::istream &in) override {return false;}
    virtual bool write(std::ostream &out) const override {return false;}
    
private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> v1_list, u1_list;
    const Eigen::Vector3d p0;
    const Eigen::Vector3d n0;
    const std::vector<Eigen::Matrix3d> R_list;
    const std::vector<Eigen::Vector3d> t_list;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    G2O_MAKE_AUTO_AD_FUNCTIONS  // Define ComputeError and linearizeOplus from operator()
};

class IBAGPREdge: public g2o::BaseUnaryEdge<20, g2o::VectorN<20>, VertexSim3>
{
public:
    /**
     * @brief Construct a new IBAGPREdge object (Auto Derivate)
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
     * @param _R_list camera relative Rotation (2nd node to 1st node)
     * @param _t_list camera relative Translation (2nd node to 1st node)
     */
    IBAGPREdge(const double &_sigma, const double &_l, const double _sigma_noise,
     const std::vector<Eigen::Vector3d> &_neigh_pts,
     const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const std::vector<double> &_u1_list, const std::vector<double> &_v1_list,
     const std::vector<Eigen::Matrix3d> &_R_list, const std::vector<Eigen::Vector3d> &_t_list,
     const bool _optimize=false, const bool _verborse=false,
     const Eigen::Vector2d &_lb=(Eigen::Vector2d() << 1e-3, 1e-3).finished(),
     const Eigen::Vector2d &_ub=(Eigen::Vector2d() << 1e3, 1e3).finished()):
     sigma(_sigma), l(_l), sigma_noise(_sigma_noise), neigh_pts(_neigh_pts),
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1_list(_u1_list), v1_list(_v1_list),
     R_list(_R_list), t_list(_t_list)
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
    bool operator()(const T* data, T* error) const{
        g2o::MatrixN<3, T> _Rcl;
        g2o::VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data); // Template Sim3 Map
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0);
        std::vector<g2o::VectorN<2, T>> _train_x;  // transformed neighbour points
        g2o::MatrixN<Eigen::Dynamic, T> _train_y;
        g2o::VectorN<2, T> _test_x = {_u0, _v0};
        _train_x.resize(neigh_pts.size());
        _train_y.resize(neigh_pts.size(), 1);
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            g2o::VectorN<3 ,T> _tf_pt = _Rcl * neigh_pts[neigh_i].cast<T>() + _tcl;
            g2o::VectorN<2, T> _uv = {_tf_pt(0) / _tf_pt(2), _tf_pt(1) / _tf_pt(2)}; /*X/Z, Y/Z*/
            _uv(0) = _fx * _uv(0) + _cx;
            _uv(1) = _fy * _uv(1) + _cy;
            _train_x[neigh_i] = _uv;
            _train_y(neigh_i) = _tf_pt(2);
        }
        TGPR gpr(sigma_noise, sigma, l, false);
        T _test_z = gpr.fit_predict<T>(_train_x, _train_y, _test_x);
        g2o::VectorN<3, T> _P0;
        _P0(0) = _test_z * (_u0 - _cx) / _fx;
        _P0(0) = _test_z * (_v0 - _cy) / _fy;
        _P0(0) = _test_z;
        const int N = u1_list.size();
        for(int i = 0; i < N; ++i)
        {
            g2o::MatrixN<3, T> _R(R_list[i].cast<T>());
            g2o::VectorN<3, T> _t(t_list[i].cast<T>());
            T _u1(u1_list[i]), _v1(v1_list[i]);
            g2o::VectorN<3, T> _P1 = _R * _P0 + _t;
            T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
            T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
            error[2*i] = _u1_obs - _u1;
            error[2*i+1] = _v1_obs - _v1;
        }
        for(int i = N; i < 10; ++i)
        {
            error[2*i] = T(0);
            error[2*i+1] = T(0);
        }
        return true;
    }

private:
    const double fx, fy, cx, cy, u0, v0;
    const std::vector<double> u1_list, v1_list;
    const std::vector<Eigen::Vector3d> neigh_pts;
    double sigma, l, sigma_noise; // We have NOT added hyperparamter adaptation during Edge Optimization YET
    const std::vector<Eigen::Matrix3d> R_list;
    const std::vector<Eigen::Vector3d> t_list;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    G2O_MAKE_AUTO_AD_FUNCTIONS  // Define ComputeError and linearizeOplus from operator()
};
