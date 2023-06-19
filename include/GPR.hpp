#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <tuple>
#include <iostream>
// #include <LBFGSB.h>

using number_t = double;

template <int N, typename T = number_t>
using VectorN = Eigen::Matrix<T, N, 1, Eigen::ColMajor>;
template <typename T = number_t>
using Vector2 = VectorN<2, T>;
template <typename T = number_t>
using VectorX = VectorN<Eigen::Dynamic, T>;

template <int N, typename T = number_t>
using MatrixN = Eigen::Matrix<T, N, N, Eigen::ColMajor>;
template <typename T = number_t>
using MatrixX = MatrixN<Eigen::Dynamic, T>;

template <typename T = number_t>
MatrixX<T> IdentityLike(const MatrixX<T> &A)
{
    assert(A.rows() == A.cols());
    MatrixX<T> I;
    I.resizeLike(A);
    I.setZero();
    for(Eigen::Index ri = 0; ri < I.rows(); ++ri)
        I(ri, ri) = T(1.0);
    return I;
}

template <typename T = number_t>
void pdist(const std::vector<Vector2<T> > &X1, const std::vector<Vector2<T> > &X2, MatrixX<T> &Dist){
    Dist.resize(X1.size(), X2.size());
    for(Eigen::Index ri = 0; ri < Dist.rows(); ++ri)
        for(Eigen::Index ci = 0; ci < Dist.cols(); ++ci)
            Dist(ri, ci) = (X1[ri] - X2[ci]).dot(X1[ri] - X2[ci]);  // sqaured
}

template <typename T = number_t>
void self_pdist(const std::vector<Vector2<T> > &X1, MatrixX<T> &Dist){
    Dist.resize(X1.size(), X1.size());
    for(Eigen::Index ri = 0; ri < Dist.rows(); ++ri)
        for(Eigen::Index ci = ri + 1; ci < Dist.cols(); ++ci)
        {
            Dist(ri, ci) = (X1[ri] - X1[ci]).dot(X1[ri] - X1[ci]);  // sqaured
            Dist(ci, ri) = Dist(ri, ci);
        }
    for(Eigen::Index ri = 0; ri < Dist.rows(); ++ri)
        Dist(ri, ri) = T(0.);
            
}

template <typename T = number_t>
MatrixX<T> rbf_kernel_2d(const std::vector<Vector2<T> > &X1, const std::vector<Vector2<T> > &X2, const T sigma, const T l){
    MatrixX<T> dist;
    pdist(X1, X2, dist);
    T sigma2 = sigma * sigma;
    T inv_l2 = T(1.0)/(l*l);
    return sigma2 * (-0.5 * inv_l2 * dist).array().exp();  
}

template <typename T = number_t>
std::tuple<MatrixX<T>, MatrixX<T>, MatrixX<T>> rbf_kernel_2d_with_grad(const std::vector<Vector2<T> > &X1, const std::vector<Vector2<T> > &X2, const T sigma, const T l){
    const int cols = X1.size(), rows = X2.size();
    MatrixX<T> dist;
    dist.resize(rows, cols);
    pdist(X1, X2, dist);
    T sigma2 = sigma * sigma;
    T inv_l2 = 1/(l*l);
    T inv_l3 = inv_l2 / l;
    MatrixX<T> K = (-0.5 * inv_l2 * dist).array().exp();  
    MatrixX<T> Jac_sigma = 2 * sigma * K;
    MatrixX<T> Jac_l = sigma2 * K * dist * inv_l3;
    return std::make_tuple(sigma2 * K, Jac_sigma, Jac_l);
}

/**
 * @brief log|A| = sum(2log(A_ii))
 * 
 * @param A 
 * @return double 
 */
double LogDet(const MatrixX<> &A){
    VectorX<> log_diag = A.diagonal().array().log();
    return 2.0 * log_diag.sum();
}

class GPRParams{
public:
    double sigma_noise = 1e-10;  // regulation factor to ensure the PSD of cholesky Decomposition
    double sigma = 10;  // kenerl param: sigma^2 * exp(-0.5/l^2 * Dist)
    double l = 10; // kenerl param: sigma^2 * exp(-0.5/l^2 * Dist)
    Eigen::Vector2d lb = {1e-4, 1e-4};  // low bound of hyperparamters for L-BFGS-B
    Eigen::Vector2d ub = {1e4, 1e4};  // upper bound of hyperparamters for L-BFGS-B
    bool optimize = true;  // whether to optimize hyperparamters
    bool verborse = false;  // print debug messages
};


template <class T = number_t>
class GPRCache{
public:
    T sigma;
    T l;
    MatrixX<T> Kff;
    MatrixX<T> L;
    VectorX<T> alpha;
    T cache_atol = 1e-8;
public:
    GPRCache(){}
    void init(int train_num){
        Kff.resize(train_num, train_num);
        L.resize(train_num, train_num);
        alpha.resize(train_num, 1);
    }
    bool allclose(const T &_sigma, const T &_l)
    {
        return (_sigma - cache_atol < sigma) && (_sigma + cache_atol > sigma) && (_l - cache_atol < l) && (_l + cache_atol > l);
    }
    void assign(const T &_sigma, const T &_l, const VectorX<T> _alpha)
    {
        sigma = _sigma;
        l = _l;
        alpha = _alpha;
    }
    void assign(const T &_sigma, const T &_l, const VectorX<T> _alpha, const MatrixX<T> &_L)
    {
        assign(_sigma, _l, _alpha);
        L = _L;
    }
    void assign(const T &_sigma, const T &_l, const VectorX<T> _alpha, const MatrixX<T> &_L, const MatrixX<T> &_Kff)
    {
        assign(_sigma, _l, _alpha, _L);
        Kff = _Kff;
    }

};

class GPRHyperLoss final : public ceres::FirstOrderFunction{
public:
    /**
     * @brief Construct a new Negative Log Marginal Likelihood Function Object
     * 
     * @param _Dist pdist matrix
     * @param _trY train_y
     * @param _sigma_noise _sigma_noise
     */
    GPRHyperLoss(const MatrixX<> &_Dist, const VectorX<> &_trY, const double &_sigma_noise):
       Dist(_Dist), trY(_trY), sigma_noise(_sigma_noise){}

    bool Evaluate(const double* parameters, double* cost, double* gradient) const{
        double sigma = parameters[0];
        double l = parameters[1];
        MatrixX<> Kff, L, Kinv;  // (N, N)
        VectorX<> alpha;  // (N, 1)
        std::tie(Kff, L ,alpha, Kinv) = compute_K_L_Alpha_Kinv(sigma, l, Dist, trY);
        int n = alpha.rows();
        cost[0] = 0.5 * (trY.dot(alpha) + LogDet(L) + n * log(2 * M_PI));
        if(gradient)
        {
            MatrixX<> dK_dsigma, dK_dl;  // (N, N)
            std::tie(dK_dsigma, dK_dl) = grad_kernel(sigma, l, Kff);
            MatrixX<> inner_term = alpha * alpha.transpose() - Kinv;  // (N, N)
            gradient[0] = -0.5 * (inner_term * dK_dsigma).trace();
            gradient[1] = -0.5 * (inner_term * dK_dl).trace();
        }
        return true;
    }

    int NumParameters() const override { return 2; }

private:
    const double sigma_noise;
    const MatrixX<> Dist;
    const VectorX<> trY;

private:
    /**
     * @brief Compute Kff, L, alpha, Kff_inv using Cached pdist and train_y
     * 
     * @tparam T typename
     * @param sigma hyper-parameter
     * @param l hyper-parameter
     * @param Dist parwise dist
     * @param train_y 
     * @return std::tuple<MatrixX<T>, MatrixX<T>, VectorX<T>, atrixX<> > Kff, L, alpha, Kff_inv
     */
    std::tuple<MatrixX<>, MatrixX<>, VectorX<>, MatrixX<> > compute_K_L_Alpha_Kinv(const double sigma ,const double l, MatrixX<> Dist, VectorX<> train_y) const
    {
        assert(Dist.rows() == train_y.rows());
        MatrixX<> Kff = computeCovariance(sigma, l, Dist);
        Kff += sigma_noise * IdentityLike(Kff);
        Eigen::LLT<MatrixX<>> llt(Kff);
        assert(llt.info() == Eigen::Success);  // if failed, enlarge the sigma_noise to ensure the PSD of Kff
        VectorX<> alpha = llt.solve(train_y);
        MatrixX<> L = llt.matrixL();
        MatrixX<> Kinv; // assign Identity first, then solve inplace
        Kinv.resizeLike(Kff);
        Kinv.setIdentity();
        llt.solveInPlace(Kinv);
        return {Kff, L, alpha, Kinv};
    }

    MatrixX<> computeCovariance(const double sigma, const double l, MatrixX<> Dist) const
    {
        return sigma * sigma * (-0.5 / (l*l) * Dist).array().exp();  // sigma^2 * exp (-0.5 * l^-2 * Dist)
    }

        /**
     * @brief gradient of RBF Kenerl
     * 
     * @param sigma 
     * @param l 
     * @param Kff 
     * @return std::tuple<double, double> dK_dsigma, dK_dl
     */
    std::tuple<MatrixX<>, MatrixX<>> grad_kernel(const double sigma, const double l, const MatrixX<> &Kff) const
    {
        Eigen::Vector2d grad; // dK_dsigma, dK_dl
        double inv_l3 = 1.0/(l*l*l);
        return {2 * sigma * Kff, Kff * Dist  * inv_l3};
    }

};

// class NLML{
// public:
//     /**
//      * @brief Construct a new Negative Log Marginal Likelihood Function Object
//      * 
//      * @param _Dist pdist matrix
//      * @param _trY train_y
//      * @param _sigma_noise _sigma_noise
//      */
//     NLML(const MatrixX<> &_Dist, const VectorX<> &_trY, const double &_sigma_noise):
//         Dist(_Dist), trY(_trY), sigma_noise(_sigma_noise){}
//      /**
//      * @brief negative log marginal likelihood
//      * 
//      * @param x 
//      * @param g 
//      * @return double 
//      */
//     double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
//     {
//         double sigma = x[0], l = x[1];
//         MatrixX<> Kff, L, Kinv;  // (N, N)
//         VectorX<> alpha;  // (N, 1)
//         std::tie(Kff, L ,alpha, Kinv) = compute_K_L_Alpha_Kinv(sigma, l, Dist, trY);
//         int n = alpha.rows();
//         double fval = 0.5 * (trY.dot(alpha) + LogDet(L) + n * log(2 * M_PI));
//         MatrixX<> dK_dsigma, dK_dl;  // (N, N)
//         std::tie(dK_dsigma, dK_dl) = grad_kernel(sigma, l, Kff);
//         MatrixX<> inner_term = alpha * alpha.transpose() - Kinv;  // (N, N)
//         g(0) = -0.5 * (inner_term * dK_dsigma).trace();  // Tr(N, N)
//         g(1) = -0.5 * (inner_term * dK_dl).trace();  // Tr(N, N)
//         return fval;
//     }

// private:
//     const double sigma_noise;
//     const MatrixX<> Dist;
//     const VectorX<> trY;
//     VectorX<> stored_alpha;

// private:
//     /**
//      * @brief Compute Kff, L, alpha, Kff_inv using Cached pdist and train_y
//      * 
//      * @tparam T typename
//      * @param sigma hyper-parameter
//      * @param l hyper-parameter
//      * @param Dist parwise dist
//      * @param train_y 
//      * @return std::tuple<MatrixX<T>, MatrixX<T>, VectorX<T>, atrixX<> > Kff, L, alpha, Kff_inv
//      */
//     std::tuple<MatrixX<>, MatrixX<>, VectorX<>, MatrixX<> > compute_K_L_Alpha_Kinv(const double sigma ,const double l, MatrixX<> Dist, VectorX<> train_y) const
//     {
//         assert(Dist.rows() == train_y.rows());
//         MatrixX<> Kff = computeCovariance(sigma, l, Dist);
//         Kff += sigma_noise * IdentityLike(Kff);
//         Eigen::LLT<MatrixX<>> llt(Kff);
//         assert(llt.info() == Eigen::Success);  // if failed, enlarge the sigma_noise to ensure the PSD of Kff
//         VectorX<> alpha = llt.solve(train_y);
//         MatrixX<> L = llt.matrixL();
//         MatrixX<> Kinv; // assign Identity first, then solve inplace
//         Kinv.resizeLike(Kff);
//         Kinv.setIdentity();
//         llt.solveInPlace(Kinv);
//         return {Kff, L, alpha, Kinv};
//     }

//     MatrixX<> computeCovariance(const double sigma, const double l, MatrixX<> Dist) const
//     {
//         return sigma * sigma * (-0.5 / (l*l) * Dist).array().exp();
//     }

//         /**
//      * @brief gradient of RBF Kenerl
//      * 
//      * @param sigma 
//      * @param l 
//      * @param Kff 
//      * @return std::tuple<double, double> dK_dsigma, dK_dl
//      */
//     std::tuple<MatrixX<>, MatrixX<>> grad_kernel(const double sigma, const double l, const MatrixX<> &Kff) const
//     {
//         Eigen::Vector2d grad; // dK_dsigma, dK_dl
//         double inv_l3 = 1.0/(l*l*l);
//         return {2 * sigma * Kff, Kff * Dist  * inv_l3};
//     }

// };


class GPR{
public:
    double sigma_noise;
    double sigma;
    double l;
    Eigen::Vector2d lb, ub;
    bool optimize;
    bool verborse;

private:
    MatrixX<> Dist;
    std::vector<Vector2<>> trX;
    VectorX<> trY;
    bool is_fit = false;

public:
    /**
     * @brief Construct a new GPR object (for hyperparameter fitting only)
     * 
     * @param params GPRParams
     */
    GPR(const GPRParams &params):
        sigma_noise(params.sigma_noise), sigma(params.sigma), l(params.l),
        lb(params.lb), ub(params.ub), optimize(params.optimize), verborse(params.verborse){}

    std::tuple<double, double> fit(const std::vector<Vector2<>> &train_x, const VectorX<> &train_y)
    {
        trX = train_x;
        trY = train_y;
        self_pdist(train_x, Dist);
        if(optimize)
        {
            double theta[2] = {sigma, l};
            ceres::GradientProblem problem(new GPRHyperLoss(Dist, trY, sigma_noise));
            ceres::GradientProblemSolver::Options options;
            options.max_num_iterations = 30;
            options.line_search_direction_type = ceres::BFGS;
            options.logging_type = ceres::SILENT; // suppress warning
            options.minimizer_progress_to_stdout = false;
            ceres::GradientProblemSolver::Summary summary;
            ceres::Solve(options, problem, theta, &summary);
            // LBFGSpp::LBFGSBParam<double> param;
            // param.epsilon = 1e-6;
            // param.max_iterations = 30;
            // param.max_linesearch = 40;
            // LBFGSpp::LBFGSBSolver<double> solver(param);
            // VectorX<> theta;
            // theta.resize(2);
            // theta(0) = sigma; theta(1) = l;
            // double fval;
            // NLML func(Dist, trY, sigma_noise);
            // int niter = solver.minimize<NLML>(func, theta, fval, lb, ub);
            // sigma = theta[0];
            // l = theta[1];
            // is_fit = true;
            if(verborse)
                std::printf("sigma: %0.4lf, l: %0.4lf\n", theta[0], theta[1]);
            sigma = theta[0];
            l = theta[1];
        }
        is_fit = true;
        return {sigma, l};
        
    }

    double predict(const VectorX<> &test_x)
    {
        assert(is_fit);
        VectorX<> Kstar = rbf_kernel_2d<>(trX, {test_x}, sigma, l);  // (N, 1)
        VectorX<> alpha;
        std::tie(std::ignore, std::ignore, alpha) = compute_K_L_Alpha(sigma, l, Dist, trY);
        return Kstar.dot(alpha);  // (1, N) * (N, 1) -> (1, 1)
    }
    
private:
    
    MatrixX<> computeCovariance(const double sigma, const double l, MatrixX<> Dist) const
    {
        return sigma * sigma * (-0.5 / (l*l) * Dist).array().exp();
    }



    /**
     * @brief Compute Kff, L, alpha using Cached pdist and train_y
     * 
     * @param sigma 
     * @param l 
     * @param Dist 
     * @param train_y 
     * @return std::tuple<MatrixX<>, MatrixX<>, VectorX<>> 
     */
    std::tuple<MatrixX<>, MatrixX<>, VectorX<>> compute_K_L_Alpha(const double sigma ,const double l, MatrixX<> Dist, VectorX<> train_y) const
    {
        assert(Dist.rows() == train_y.rows());
        MatrixX<> Kff = computeCovariance(sigma, l, Dist);
        Kff += sigma_noise * IdentityLike(Kff);
        Eigen::LLT<MatrixX<>> llt(Kff);
        assert(llt.info() == Eigen::Success);  // if failed, enlarge the sigma_noise to ensure the PSD of Kff
        VectorX<> alpha = llt.solve(train_y);
        MatrixX<> L = llt.matrixL();
        return {Kff, L, alpha};
    }


};


class TGPR{
public:
    double sigma_noise;
    double sigma;
    double l;
    bool verborse;

public:
    /**
     * @brief Construct a new GPR object (only support double type operations)
     * 
     * @param params GPRParams
     */
    TGPR(const double &_sigma_noise, const double &_sigma, const double &_l, const bool _verborse):
        sigma_noise(_sigma_noise), sigma(_sigma), l(_l), verborse(_verborse){}

    template <typename T>
    T fit_predict(const std::vector<Vector2<T>> &train_x, const VectorX<T> &train_y, const Vector2<T> &test_x){
        int N = train_x.size();
        MatrixX<T> Dist;
        self_pdist<T>(train_x, Dist);
        MatrixX<T> Kff;
        MatrixX<T> L;
        VectorX<T> alpha;
        T _sigma(sigma), _l(l);
        std::tie(Kff, L, alpha) = compute_K_L_Alpha<T>(_sigma, _l, Dist, train_y);
        VectorX<T> Kstar = rbf_kernel_2d<T>(train_x, {test_x}, _sigma, _l);  // (N, 1)
        T mu = Kstar.transpose() * alpha; // (1, N) * (N, 1) -> (1,)
        return mu;
    }

private:    
    template <typename T>
    MatrixX<T> computeCovariance(const T sigma, const T l, MatrixX<T> Dist) const
    {
        return sigma * sigma * (-0.5 / (l*l) * Dist).array().exp();
    }

    /**
     * @brief Compute Kff, L, alpha using Cached pdist and train_y
     * 
     * @tparam T typename
     * @param sigma hyper-parameter
     * @param l hyper-parameter
     * @param Dist parwise dist
     * @param train_y 
     * @return std::tuple<MatrixX<T>, MatrixX<T>, VectorX<T>> Kff, L, alpha
     */
    template <typename T>
    std::tuple<MatrixX<T>, MatrixX<T>, VectorX<T>> compute_K_L_Alpha(const T sigma ,const T l, MatrixX<T> Dist, VectorX<T> train_y) const
    {
        assert(Dist.rows() == train_y.rows());
        MatrixX<T> Kff = computeCovariance<T>(sigma, l, Dist);
        Kff += T(sigma_noise) * IdentityLike(Kff);
        Eigen::LLT<MatrixX<T>> llt(Kff);
        assert(llt.info() == Eigen::Success);  // if failed, enlarge the sigma_noise to ensure the PSD of Kff
        VectorX<T> alpha = llt.solve(train_y);
        MatrixX<T> L = llt.matrixL();
        return {Kff, L, alpha};
    }

};