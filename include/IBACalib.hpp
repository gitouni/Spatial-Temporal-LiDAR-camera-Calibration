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

class IBAPlaneEdge: public g2o::BaseUnaryEdge<2, g2o::Vector2, g2o::VertexSE3Expmap>
{
public:
    /**
     * @brief Construct a new IBAPlaneEdge Object (Analytical Derivate)
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
     * @param _n0 normal of _p0 (in the LIDAR Coord System)
     * @param _R camera relative Rotation (R_21)
     * @param _t camera relative Translation (t_21)
     */
    IBAPlaneEdge(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const double &_u1, const double &_v1,
     const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_t):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1(_u1), v1(_v1),
     p0(_p0), n0(_n0), R(_R), t(_t){};
    virtual void computeError() override {
        const g2o::VertexSE3Expmap *v = static_cast<g2o::VertexSE3Expmap *> (_vertices[0]);
        g2o::SE3Quat curr_se3 = v->estimate();  // TCL
        g2o::Vector3 p0c = curr_se3 * p0;
        g2o::Vector3 n0c = curr_se3.rotation() * n0;  // normal vector is only affected by rotation
        double Cxz = (u0-cx)/fx;
        double Cyz = (v0-cy)/fy;
        double Z0 = n0c.dot(p0c) / (Cxz*n0c(0) + Cyz*n0c(1) + n0c(2));
        double X0 = Cxz * Z0;
        double Y0 = Cyz * Z0;
        g2o::Vector3 P0(X0, Y0, Z0);
        g2o::Vector3 P1 = R * P0 + t;
        double u1_obs = fx*P1(0)/P1(2) + cx;
        double v1_obs = fy*P1(1)/P1(2) + cy;
        _error << u1_obs - u1, v1_obs - v1;
    }
    virtual void linearizeOplus() override {
        const g2o::VertexSE3Expmap *v = static_cast<g2o::VertexSE3Expmap *> (_vertices[0]);
        g2o::SE3Quat curr_se3 = v->estimate();  // TCL
        g2o::Vector6 curr_se3_log = curr_se3.log(); // [omega, t]
        g2o::Vector3 p0c = curr_se3 * p0;
        g2o::Vector3 n0c = curr_se3.rotation() * n0;  // normal vector is only affected by rotation
        double Cxz = (u0-cx)/fx;
        double Cyz = (v0-cy)/fy;
        double numZ0 = n0c.dot(p0c);
        Eigen::Vector3d Czvec(Cxz, Cyz, 1);
        double denZ0 = Czvec.dot(n0c);
        double Z0 = numZ0 / denZ0;
        double X0 = Cxz * Z0;
        double Y0 = Cyz * Z0;
        g2o::Vector3 P0(X0, Y0, Z0);
        g2o::Vector3 P1 = R * P0 + t;

        Eigen::Matrix<double, 2, 3> duv_dP1; 
        double X1 = P1(0), Y1 = P1(1), invZ1 = 1/P1(2), invZ1SQ = invZ1 * invZ1;
        duv_dP1 << -fx*invZ1, 0, fx*X1*invZ1SQ, 0, -fy*invZ1, fy*Y1*invZ1SQ; // (2, 3)

        Eigen::Matrix3d K;
        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        Eigen::Matrix3d dP1_dP0 = K * R;  // P1 = K*(RP+t), (3, 3)

        double x0 = p0c(0), y0 = p0c(1), z0 = p0c(2);
        Eigen::Matrix<double, 1, 6> dZ0_dp0cn0c;
        dZ0_dp0cn0c.leftCols(3) = (n0c / denZ0).transpose();
        dZ0_dp0cn0c.rightCols(3) = ((p0*denZ0 - numZ0*Czvec)/(denZ0*denZ0)).transpose();
        Eigen::Matrix<double, 3, 6> dP0_dp0n0c = Czvec * dZ0_dp0cn0c; // (3, 1) x (1, 6) -> (3, 6)

        Eigen::Matrix<double, 3, 6> dp0c_dx;
        dp0c_dx.leftCols(3) = -1 * g2o::skew(curr_se3 * p0c);   // (-Rp+t)^
        dp0c_dx.rightCols(3) = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 3, 6> dn0c_dx;
        dn0c_dx.leftCols(3) = -1 * g2o::skew(curr_se3.rotation() * n0c); // (-Rp)^
        dn0c_dx.rightCols(3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 6, 6> dp0n0c_dx;
        dp0n0c_dx << dp0c_dx, dn0c_dx; // (3, 6) U (3, 6) -> (6, 6)
        _jacobianOplusXi << duv_dP1 * dP1_dP0 * dP0_dp0n0c * dp0n0c_dx; // (2, 3) x (3, 3) x (3, 6) x (6, 6) -> (2, 6)
    }
    virtual bool read(std::istream &in) override {return false;}
    virtual bool write(std::ostream &out) const override {return false;}
    
private:
    const double fx, fy, cx, cy, u0, v0, u1, v1;
    const Eigen::Vector3d p0;
    const Eigen::Vector3d n0;
    const Eigen::Matrix3d R;  // relative Rotataion between KeyFrames
    const Eigen::Vector3d t;  // relative Translation between KeyFrames
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class IBAPlaneEdgeAD: public g2o::BaseUnaryEdge<2, g2o::Vector2, VertexSim3>
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
     * @param _n0 normal of _p0 (in the LIDAR Coord System)
     * @param _R camera relative Rotation (2nd node to 1st node)
     * @param _t camera relative Translation (2nd node to 1st node)
     */
    IBAPlaneEdgeAD(const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const double &_u1, const double &_v1,
     const Eigen::Vector3d &_p0, const Eigen::Vector3d &_n0, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_t):
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1(_u1), v1(_v1),
     p0(_p0), n0(_n0), R(_R), t(_t){};
    

    template <typename T>
    bool operator()(const T* data, T* error) const{
        g2o::MatrixN<3, T> _Rcl;
        g2o::VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data); // Template Sim3 Map
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0), _u1(u1), _v1(v1);
        g2o::VectorN<3, T> _p0 = p0.cast<T>();
        g2o::VectorN<3, T> _n0 = n0.cast<T>();
        g2o::MatrixN<3, T> _R = R.cast<T>();
        g2o::VectorN<3 ,T> _t = t.cast<T>() * _s;
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
        error[0] = _u1_obs - _u1;
        error[1] = _v1_obs - _v1;
        return true;
    }
    
    virtual bool read(std::istream &in) override {return false;}
    virtual bool write(std::ostream &out) const override {return false;}
    
private:
    const double fx, fy, cx, cy, u0, v0, u1, v1;
    const Eigen::Vector3d p0;
    const Eigen::Vector3d n0;
    const Eigen::Matrix3d R;
    const Eigen::Vector3d t;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    G2O_MAKE_AUTO_AD_FUNCTIONS  // Define ComputeError and linearizeOplus from operator()
};

class IBAGPREdge: public g2o::BaseUnaryEdge<2, g2o::Vector2, VertexSim3>
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
     * @param _u1 second correspondence point x
     * @param _v1 second correspondence point y
     * @param _R camera relative Rotation (2nd node to 1st node)
     * @param _t camera relative Translation (2nd node to 1st node)
     */
    IBAGPREdge(const double &_sigma, const double &_l, const double _sigma_noise,
     const std::vector<Eigen::Vector3d> &_neigh_pts,
     const double &_fx, const double &_fy, const double &_cx, const double &_cy,
     const double &_u0, const double &_v0, const double &_u1, const double &_v1,
     const Eigen::Matrix3d &_R, const Eigen::Vector3d &_t,
     const bool _optimize=false, const bool _verborse=false,
     const Eigen::Vector2d &_lb={1e-3, 1e-3}, const Eigen::Vector2d &_ub={1e3, 1e3}):
     sigma(_sigma), l(_l), sigma_noise(_sigma_noise), neigh_pts(_neigh_pts),
     fx(_fx), fy(_fy), cx(_cx), cy(_cy),
     u0(_u0), v0(_v0), u1(_u1), v1(_v1),
     R(_R), t(_t)
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
     }

    template <typename T>
    bool operator()(const T* data, T* error) const{
        g2o::MatrixN<3, T> _Rcl;
        g2o::VectorN<3, T> _tcl;
        T _s;
        std::tie(_Rcl, _tcl, _s) = Sim3Exp<T>(data); // Template Sim3 Map
        g2o::MatrixN<3, T> _R(R.cast<T>());
        g2o::VectorN<3, T> _t(t.cast<T>());
        T _fx(fx), _fy(fy), _cx(cx), _cy(cy), _u0(u0), _v0(v0), _u1(u1), _v1(v1);
        std::vector<g2o::VectorN<2, T>> _train_x;  // transformed neighbour points
        g2o::MatrixN<Eigen::Dynamic, T> _train_y;
        g2o::VectorN<2, T> _test_x = {_u0, _v0};
        _train_x.reserve(neigh_pts.size());
        _train_y.resize(neigh_pts.size(), 1);
        for(std::size_t neigh_i = 0; neigh_i < neigh_pts.size(); ++neigh_i)
        {
            g2o::VectorN<3 ,T> _tf_pt = _Rcl * neigh_pts[neigh_i].cast<T>() + _tcl;
            g2o::VectorN<2, T> _uv = {_tf_pt(0) / _tf_pt(2), _tf_pt(1) / _tf_pt(2)}; /*X/Z, Y/Z*/
            _uv(0) = _fx * _uv(0) + _cx;
            _uv(1) = _fy * _uv(1) + _cy;
            _train_x.push_back(_uv);
            _train_y(neigh_i) = _tf_pt(2);
        }
        TGPR gpr(sigma_noise, sigma, l, false);
        T _test_z = gpr.fit_predict<T>(_train_x, _train_y, _test_x);
        g2o::VectorN<3, T> _P0;
        _P0(0) = _test_z * (_u0 - _cx) / _fx;
        _P0(0) = _test_z * (_v0 - _cy) / _fy;
        _P0(0) = _test_z;
        g2o::VectorN<3, T> _P1 = _R * _P0 + _t;
        T _u1_obs = _fx*_P1(0)/_P1(2) + _cx;
        T _v1_obs = _fy*_P1(1)/_P1(2) + _cy;
        error[0] = _u1_obs - _u1;
        error[1] = _v1_obs - _v1;
        return true;
    }

private:
    const double fx, fy, cx, cy, u0, v0, u1, v1;
    const std::vector<Eigen::Vector3d> neigh_pts;
    double sigma, l, sigma_noise; // We have NOT added hyperparamter adaptation during Edge Optimization YET
    const Eigen::Matrix3d R;
    const Eigen::Vector3d t;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    G2O_MAKE_AUTO_AD_FUNCTIONS  // Define ComputeError and linearizeOplus from operator()
};
