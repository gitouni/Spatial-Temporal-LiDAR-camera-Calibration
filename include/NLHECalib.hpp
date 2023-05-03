#include <Eigen/Dense>
#include <vector>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <unordered_set>
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/core/auto_differentiation.h"
#include <algorithm>

namespace Eigen{
    Eigen::Matrix3d skew(Eigen::Vector3d vec){
        return (Eigen::Matrix3d() << 0, vec(2), -vec(1), -vec(2), 0, vec(0), vec(1), -vec(0), 0).finished();
    }
}

class VertexPose : public g2o::BaseVertex<7, g2o::Vector7> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() override {
        _estimate = (g2o::Vector7() << 0, 0, 0, 0, 0, 0, 1.0).finished();
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        g2o::Vector7::ConstMapType updateVec(update);
        _estimate += updateVec;
    }

    virtual bool read(std::istream &in) override {return false;}

    virtual bool write(std::ostream &out) const override {return false;}

    std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> toPose() const{
        Eigen::Vector3d rotvec(_estimate.head<3>());
        Eigen::Vector3d translation(_estimate.segment<3>(3, 3));
        double scale = _estimate(6);
        double rotvec_norm = rotvec.norm();
        Eigen::AngleAxisd ax(rotvec_norm, rotvec/rotvec_norm);
        return {ax.toRotationMatrix(), translation, scale};
    }
};

class EdgeHE : public g2o::BaseUnaryEdge<3, g2o::Vector3, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeHE(const Eigen::Isometry3d &Ta, const Eigen::Isometry3d &Tb, const double weight): _Ta(Ta), _Tb(Tb), _weight(weight){}
    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Eigen::Matrix3d Rab;
        Eigen::Vector3d tab;
        double s;
        std::tie(Rab, tab, s) = v->toPose();
        Eigen::Vector4d tsab;  // [tab, s]
        tsab.block<3, 1>(0, 0) = tab;
        tsab(3) = s;
        Eigen::AngleAxisd axisTa(_Ta.rotation()), axisTb(_Tb.rotation());
        Eigen::Vector3d errRotVec = Rab * (axisTb.angle() * axisTb.axis()) - (axisTa.angle() * axisTa.axis()); 
        Eigen::Matrix<double, 3, 4> A;
        Eigen::Vector3d b;
        A.block<3, 3>(0, 0) = _Ta.rotation() - Eigen::Matrix3d::Identity();
        A.block<3, 1>(0, 3) = _Ta.translation();
        b = Rab * _Tb.translation();
        Eigen::Vector3d errTran = A * tsab - b;
        _error << _weight * (errRotVec + errTran);
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Eigen::Matrix3d Rab;
        Eigen::Vector3d tab;
        double s;
        std::tie(Rab, tab, s) = v->toPose();
        Eigen::Vector4d tsab;  // [tab, s]
        tsab.block<3, 1>(0, 0) = tab;
        tsab(3) = s;
        Eigen::AngleAxisd axisTb(_Tb.rotation());
        Eigen::Matrix3d JacobianRotVec = -1.0 * Eigen::skew(Rab * (axisTb.angle() * axisTb.axis())); // -(Rp)^ or (-Rp)^
        Eigen::Matrix<double, 3, 4> JacobianTran;
        JacobianTran.block<3, 3>(0, 0) = _Ta.rotation() - Eigen::Matrix3d::Identity();
        JacobianTran.block<3, 1>(0, 3) = _Ta.translation();
        _jacobianOplusXi << _weight * JacobianRotVec, _weight * JacobianTran;  // (3,3) (3,4) -> concat -> (3,7)
    }

    void updateWeight(double weight){
        _weight = weight;
    }

    double returnSquaredWeight(){
        return _weight * _weight;
    }

    virtual bool read(std::istream &in) override {return false;}

    virtual bool write(std::ostream &out) const override {return false;}
private:
    Eigen::Isometry3d _Ta, _Tb;
    double _weight;
};


class EdgeRegulation : public g2o::BaseUnaryEdge<3, g2o::Vector3, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeRegulation(){}
    // virtual void computeError() override {
    //     const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    //     g2o::Sim3 Tab(v->estimate());
    //     Eigen::Vector3d translation = Tab.translation();
    //     _error << translation;
    // }

    // virtual void linearizeOplus() override {
    //     const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    //     Eigen::Matrix<double, 3, 7> Jacobian;
    //     Jacobian.setZero();
    //     Jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    //     _jacobianOplusXi << Jacobian;  // (3,3) (3,4) -> concat -> (3,7)
    // }

    virtual bool read(std::istream &in) override {return false;}

    virtual bool write(std::ostream &out) const override {return false;}
    template <typename T>
    bool operator()(const T* params, T* error) const {
        error[0] = params[3];
        error[1] = params[4];
        error[2] = params[5];
        return true;
    }
    G2O_MAKE_AUTO_AD_FUNCTIONS
};

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> HECalibRobustKernelg2o(const std::vector<Eigen::Isometry3d> &vTa, const std::vector<Eigen::Isometry3d> &vTb,
    const Eigen::Matrix3d &initialRotation, const Eigen::Vector3d &initialTranslation, const double &initialScale,
    const bool regulation=true, const double regulation_ratio= 0.005, const bool verbose=true){

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<7, 3>> BlockSolverType; 
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    auto solver = new g2o::OptimizationAlgorithmDogleg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(verbose);       // 打开调试输出
    VertexPose *v(new VertexPose);
    Eigen::AngleAxisd ax(initialRotation);
    Eigen::Vector3d rotvec = ax.angle() * ax.axis();
    g2o::Vector7 initialValue;
    initialValue.head<3>() = rotvec;
    initialValue.segment<3>(3, 3) = initialTranslation;
    initialValue(6) = initialScale;
    v->setEstimate(initialValue);
    v->setId(0);
    optimizer.addVertex(v);
    for(std::size_t i = 0; i < vTa.size(); ++i){
        EdgeHE* edge = new EdgeHE(vTa[i], vTb[i], 1.0);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setInformation(Eigen::Matrix3d::Identity());
        g2o::RobustKernelHuber* rk(new g2o::RobustKernelHuber);
        rk->setDelta(0.1);
        edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
    }
    if(regulation){
        EdgeRegulation* regedge = new EdgeRegulation();
        regedge->setId(vTa.size());
        regedge->setVertex(0, v);
        double ratio = vTa.size() * regulation_ratio;
        regedge->setInformation(Eigen::Matrix3d::Identity() * ratio);
        optimizer.addEdge(regedge);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    Eigen::Matrix3d RCL;
    Eigen::Vector3d tCL;
    double scale;
    std::tie(RCL, tCL, scale) = v->toPose();
    if(verbose){
        std::vector<double> chi2List;
        chi2List.reserve(optimizer.edges().size());
        double chi2Mean = 0, RegChi2 = 0;
        for(auto it = optimizer.edges().begin(); it != optimizer.edges().end(); ++it){
            if((*it)->id() < (int) vTa.size()){
                EdgeHE* edge = dynamic_cast<EdgeHE*> (*it);
                chi2List.push_back((double)edge->chi2());
                chi2Mean += (double)edge->chi2();
            }else{
                EdgeRegulation* edge = dynamic_cast<EdgeRegulation*> (*it);
                RegChi2 = (double)edge->chi2();
            }
        }
        std::sort(chi2List.begin(),chi2List.end());
        std::cout << "Squared Error:\n";
        std::cout << "Max: " << chi2List[chi2List.size()-1] << std::endl;
        std::cout << "Min: " << chi2List[0] << std::endl;
        std::cout << "Median: " << chi2List[chi2List.size()/2] << std::endl;
        std::cout << "Mean: " << chi2Mean/chi2List.size() << std::endl;
        if(regulation){
            std::cout << "Regulation Squared Error: " << RegChi2 << std::endl;
        }
    }
    return {RCL, tCL, scale};
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> HECalibLPg2o(const std::vector<Eigen::Isometry3d> &vTa, const std::vector<Eigen::Isometry3d> &vTb,
    const Eigen::Matrix3d &initialRotation, const Eigen::Vector3d &initialTranslation, const double &initialScale, const unsigned short in_max_iter = 5,
    const double mu0 = 64, const double divid_factor = 1.4, const double min_mu = 1e-4, const unsigned short ex_max_iter = 20,
    const bool regulation=true, const double regulation_ratio= 0.005, const bool verbose=true){

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<7, 3>> BlockSolverType; 
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(verbose);       // 打开调试输出
    VertexPose *v(new VertexPose);
    Eigen::AngleAxisd ax(initialRotation);
    Eigen::Vector3d rotvec = ax.angle() * ax.axis();
    g2o::Vector7 initialValue;
    initialValue.head<3>() = rotvec;
    initialValue.segment<3>(3, 3) = initialTranslation;
    initialValue(6) = initialScale;
    v->setEstimate(initialValue);
    v->setId(0);
    optimizer.addVertex(v);
    for(std::size_t i = 0; i < vTa.size(); ++i){
        EdgeHE* edge(new EdgeHE(vTa[i], vTb[i], 1.0));
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setInformation(Eigen::Matrix<double, 3, 3>::Identity());
        optimizer.addEdge(edge);
    }
    if(regulation){
        EdgeRegulation* regedge = new EdgeRegulation();
        regedge->setId(vTa.size());
        regedge->setVertex(0, v);
        double ratio = vTa.size() * regulation_ratio;
        regedge->setInformation(Eigen::Matrix3d::Identity() * ratio);
        optimizer.addEdge(regedge);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    double mu = mu0;
    for(unsigned short exiter = 0; exiter < ex_max_iter; ++exiter){
        for(auto it = optimizer.edges().begin(); it != optimizer.edges().end(); ++it){
            if((*it)->id() < (int) vTa.size()){
                EdgeHE* edge = dynamic_cast<EdgeHE*> (*it);
                double e2 = (double)(edge->chi2());
                double w = mu/(mu+e2);
                edge->setInformation(w*w*Eigen::Matrix3d::Identity());
            }
        }
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        if(mu >= min_mu)
            mu /= divid_factor;
    }
    Eigen::Matrix3d RCL;
    Eigen::Vector3d tCL;
    double scale;
    std::tie(RCL, tCL, scale) = v->toPose();
    if(verbose){
        std::vector<double> chi2List;
        chi2List.reserve(optimizer.edges().size());
        double chi2Mean = 0, RegChi2 = 0;
        for(auto it = optimizer.edges().begin(); it != optimizer.edges().end(); ++it){
            if((*it)->id() < (int) vTa.size()){
                EdgeHE* edge = dynamic_cast<EdgeHE*> (*it);
                chi2List.push_back((double)edge->chi2());
                chi2Mean += (double)edge->chi2();
            }else{
                EdgeRegulation* edge = dynamic_cast<EdgeRegulation*> (*it);
                RegChi2 = (double)edge->chi2();
            }
        }
        std::sort(chi2List.begin(),chi2List.end());
        std::cout << "Squared Error:\n";
        std::cout << "Max: " << chi2List[chi2List.size()-1] << std::endl;
        std::cout << "Min: " << chi2List[0] << std::endl;
        std::cout << "Median: " << chi2List[chi2List.size()/2] << std::endl;
        std::cout << "Mean: " << chi2Mean/chi2List.size() << std::endl;
        if(regulation){
            std::cout << "Regulation Squared Error: " << RegChi2 << std::endl;
        }
    }
    return {RCL, tCL, scale};
}