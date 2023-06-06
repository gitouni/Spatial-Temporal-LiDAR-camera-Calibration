#include <Eigen/Dense>
#include <LBFGSB.h>
#include <tuple>

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
void pdist(const std::vector<Vector2<T> > &X1, const std::vector<Vector2<T> > &X2, MatrixX<T> &Dist){
    Dist.resize(X1.size(), X2.size());
    for(Eigen::Index ri = 0; ri < Dist.rows(); ++ri)
        for(Eigen::Index ci = 0; ci < Dist.cols(); ++ci)
            Dist(ri, ci) = (X1[ci] - X2[ri]).dot(X1[ci] - X2[ri]);  // sqaured
}


template <typename T = number_t>
MatrixX<T> rbf_kernel_2d(const std::vector<Vector2<T> > &X1, const std::vector<Vector2<T> > &X2, const T sigma, const T l){
    const int cols = X1.size(), rows = X2.size();
    MatrixX<T> dist;
    dist.resize(rows, cols);
    pdist(X1, X2, dist);
    T sigma2 = sigma * sigma;
    T inv_l2 = 1/(l*l);
    T inv_l3 = inv_l2 / l;
    return sigma2 * (-0.5 * inv_l2 * dist).exp();  
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
    MatrixX<T> K = (-0.5 * inv_l2 * dist).exp();  
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
    VectorX<> log_diag = A.diagonal().log();
    return 2*log_diag.sum();
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
    GPRCache(){};
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
    GPRCache<> Cache;
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

    void fit(const std::vector<Vector2<>> &train_x, const VectorX &train_y)
    {
        trX = train_x;
        trY = train_y;
        pdist(train_x, train_x, Dist);
        if(optimize)
        {
            LBFGSpp::LBFGSParam<double> param;
            param.epsilon = 1e-6;
            param.max_iterations = 30;
            LBFGSpp::LBFGSBSolver<double> solver(param);
            Eigen::Vector2d theta = {sigma, l};
            double fval;
            int niter = solver.minimize<double>(&optmize_func, theta, fval, lb, ub);
            sigma = theta[0];
            l = theta[1];
            is_fit = true;
            if(verborse)
            {
                char msg[100];
                sprintf(msg, "optiter:%d, sigma: %0.4lf, l: %0.4lf\n", niter, theta[0], theta[1]);
                std::cout << msg;
            }
        }
        
    }

    double predict(const VectorX &test_x)
    {
        assert(is_fit);
        MatrixX<> Kstar = rbf_kernel_2d<>(trX, {test_x}, sigma, l);  // (N, 1)
        VectorX<> alpha;
        if(Cache.allclose(sigma, l)){
            alpha = Cache.alpha;
        }else
        {
            std::tie(_, _, alpha) = compute_K_L_Alpha(sigma, l, Dist, trY);
        }
        return Kstar.transpose() * alpha;  // (1, N) * (N, 1) -> (1, 1)
    }
    
private:
     /**
     * @brief negative log marginal likelihood
     * 
     * @param x 
     * @param g 
     * @return double 
     */
    double optmize_func(const Eigen::VectorXd &x, Eigen::VectorXd &g)
    {
        double sigma_f = x[0], l = x[1];
        MatrixX<> Kff, L, Kinv;  // (N, N)
        VectorX<> alpha;  // (N, 1)
        std::tie(Kff, L ,alpha, Kinv) = compute_K_L_Alpha_Kinv(sigma_f, l, Dist, trY);
        double fval = 0.5 * (trY.dot(alpha) + LogDet(L) + n * log(2 * M_PI));
        MatrixX<> dK_dsigma, dK_dl;  // (N, N)
        std::tie(dK_dsigma, dK_dl) = grad_kernel(sigma, l, Kff);
        MatrixX<> inner_term = alpha * alpha.transpose() - Kinv;  // (N, N)
        g(0) = 0.5 * (inner_term * dK_dsigma).trace();  // Tr(N, N)
        g(1) = 0.5 * (inner_term * dK_dl).trace();  // Tr(N, N)
        return fval;
    }
    MatrixX<> computeCovariance(const double sigma, const double l, MatrixX<> Dist) const
    {
        return sigma * sigma * (-0.5 / (l*l) * Dist).exp();
    }

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
        MatrixX<> Kff = computeCovariance<T>(sigma, l, Dist);
        Kff += T(sigma_noise) * MatrixX<>::Identity();
        Eigen::LLT<MatrixX<>> llt(Kff);
        assert(llt.info() == Eigen::Success);  // if failed, enlarge the sigma_noise to ensure the PSD of Kff
        VectorX<> alpha = llt.solve(train_y);
        MatrixX<> L = llt.matrixL();
        MatrixX<> Kinv; // assign Identity first, then solve inplace
        Kinv.resize(Kff.rows(), Kff.cols());
        Kinv.setIdentity();
        llt.solveInPlace(Kinv);
        Cache.assign(sigma, l, alpha);  // only need cached alpha for predict
        return {Kff, L, alpha, Kinv};
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
        MatrixX<> Kff = computeCovariance<T>(sigma, l, Dist);
        Kff += T(sigma_noise) * MatrixX<>::Identity();
        Eigen::LLT<MatrixX<>> llt(Kff);
        assert(llt.info() == Eigen::Success);  // if failed, enlarge the sigma_noise to ensure the PSD of Kff
        VectorX<> alpha = llt.solve(train_y);
        MatrixX<> L = llt.matrixL();
        Cache.assign(sigma, l, alpha);  // only need cached alpha for predict
        return {Kff, L, alpha};
    }

    /**
     * @brief gradient of RBF Kenerl
     * 
     * @param sigma 
     * @param l 
     * @param Kff 
     * @return std::tuple<double, double> dK_dsigma, dK_dl
     */
    std::tuple<MatrixX<>, MatrixX<>> grad_kernel(const double sigma, const double l, const MatrixX<> &Kff)
    {
        Eigen::Vector2d grad; // dK_dsigma, dK_dl
        double inv_l3 = 1.0/(l*l*l);
        return {2 * sigma * Kff, Kff * Dist  * inv_l3};
    }
};

template <class T>
class TGPR{
public:
    T sigma_noise;
    T sigma;
    T l;
    bool verborse;

public:
    /**
     * @brief Construct a new GPR object (only support double type operations)
     * 
     * @param params GPRParams
     */
    TGPR(const GPRParams &params, const MatrixX<T> &_Dist):
        sigma_noise(params.sigma_noise), sigma(params.sigma), l(params.l),
        verborse(params.verborse){}

    T fit_predict(const std::vector<Vector2<T>> &train_x, const VectorX<T> &train_y, const Vector2<T> &test_x){
        int N = train_x.size();
        MatrixX<T> Dist;
        pdist<T>(train_x, train_x, Dist);
        MatrixX<T> Kff;
        MatrixX<T> L;
        VectorX<T> alpha;
        std::tie(Kff, L, alpha) = compute_K_L_Alpha<T>(sigma, l, Dist, train_y);
        VectorX<T> Kstar = rbf_kernel_2d<T>(train_x, {test_x}, sigma, l);  // (N, 1)
        T mu = Kstar.transpose() * alpha; // (1, N) * (N, 1) -> (1,)
        return mu;
    }

private:    
    MatrixX<T> computeCovariance(const T sigma, const T l, MatrixX<T> Dist) const
    {
        return sigma * sigma * (-0.5 / (l*l) * Dist).exp();
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
    std::tuple<MatrixX<T>, MatrixX<T>, VectorX<T>> compute_K_L_Alpha(const T sigma ,const T l, MatrixX<T> Dist, VectorX<T> train_y) const
    {
        assert(Dist.rows() == train_y.rows());
        MatrixX<T> Kff = computeCovariance<T>(sigma, l, Dist);
        int N = Kff.cols();
        Kff += T(sigma_noise) * MatrixN<N, T>::Identity();
        Eigen::LLT<MatrixN<N, T>> llt(Kff);
        assert(llt.info() == Eigen::Success);  // if failed, enlarge the sigma_noise to ensure the PSD of Kff
        VectorN<N, T> alpha = llt.solve(train_y);
        MatrixN<N, T> L = llt.matrixL();
        return {Kff, L, alpha};
    }

};