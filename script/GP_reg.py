import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

def cholesky_solve(A:np.ndarray, b:np.ndarray, return_cholesky_factor=False):
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    if not return_cholesky_factor:
        return np.linalg.solve(L.T, y)
    else:
        return np.linalg.solve(L.T, y), L 

def cholesky_inv(A:np.ndarray, return_cholesky_factor=False):
    I = np.eye(A.shape[0])
    return cholesky_solve(A, b=I, return_cholesky_factor=return_cholesky_factor)
    
def qr_solve(A, b):
    Q, R = np.linalg.qr(A)
    QT_B = Q.T @ b
    return np.linalg.solve(R, QT_B)

def get_sklearn_gpr(constant_value=1.,constant_value_bounds=(1e-5, 1e5), length_scale=1., length_scale_bounds=(1e-5,1e5)):
    kernel = ConstantKernel(constant_value=constant_value, constant_value_bounds=constant_value_bounds) * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
    gpr = GaussianProcessRegressor(kernel=kernel)
    return gpr

def pdist(x1:np.ndarray, x2:np.ndarray):
    if(len(x1.shape)==len(x2.shape)==1):
        return (x1[:,None]-x2[None,:])**2
    # x1,x2 : N,d
    dist_matrix:np.ndarray = (x1[:,None,...]-x2[None,...])**2  # (1,N,d) - (N,1,d) -> (N,N,d)
    dist_matrix = np.sum(dist_matrix,axis=-1)  # (N,N,d) -> (N,N)
    return dist_matrix

def rbf_kernel(x1:np.ndarray, x2:np.ndarray, l=1.0, sigma=1.0)->np.ndarray:
    """More efficient approach."""
    dist_matrix = pdist(x1,x2)
    return sigma ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

class LegacyGPR:
    
    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize
       
    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        
         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()
                
        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]], 
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]
        
        self.is_fit = True
    
    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return
        
        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)
    

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.l_bound = [1e-2, 1e2]
        self.sigma_bound = [1e-2, 1e2]
        self.params = {"l": 10, "sigma_f": 10, "noise":1e-10}
        self.optimize = optimize
        self.dist = None
        self.log2pi = np.log(2*np.pi)
        self.theta_cache = np.array([self.params['sigma_f'], self.params['l']])
        self.K_cache = None
        self.L_cache = None
        self.alpha_cache = None
        self.cache_atol = 0

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.dist = pdist(self.train_X, self.train_X)
        self.n = self.train_X.shape[0]
        if self.optimize:
            res = minimize(self._log_marginal_likelihood, [self.params["sigma_f"],self.params['l']],
                           method='L-BFGS-B',jac=self._grad_log_marginal_likelihood,bounds=[self.sigma_bound, self.l_bound])
            self.params['sigma_f'], self.params['l'] = res.x[0], res.x[1]
        self.is_fit = True
        
    def predict(self, x_test:np.ndarray, return_cov=False):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return
        _, L, alpha = self.K_L_alpha(self.theta_cache)
        K_trans = self.kernel(self.train_X, x_test)
        fmean = np.dot(K_trans.T, alpha)
        if not return_cov:
            return fmean
        else:
            v:np.ndarray = np.linalg.solve(L, K_trans)
            f_cov = self.kernel(x_test, x_test) - np.dot(v.T, v)
            return fmean, f_cov
    
    def _log_marginal_likelihood(self, theta):
        _, L, alpha = self.K_L_alpha(theta)
        return 0.5 * np.dot(self.train_y.T, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * self.n * self.log2pi
    
    def _grad_log_marginal_likelihood(self, theta):
        K, _, alpha = self.K_L_alpha(theta)
        dK_dsigma, dK_dl = self.grad_Kff(theta[0], theta[1], K)
        Kinv = cholesky_inv(K)
        tmp = alpha.reshape(-1,1) @ alpha.reshape(1,-1) - Kinv
        dC_dsigma = 0.5 * np.trace(tmp @ dK_dsigma)
        dC_dl = 0.5 * np.trace(tmp @ dK_dl)
        return -np.array([dC_dsigma, dC_dl])
    
    def grad_Kff(self, sigma, l, K=None):
        if K is None:
            Kff = self.Kff(sigma, l)
        else:
            Kff = K
        dK_dl = Kff * self.dist / l**3
        dK_dsigma = 2 * sigma * Kff
        return dK_dsigma, dK_dl
        

    def kernel(self, x1, x2) -> np.ndarray:
        dist_matrix = pdist(x1, x2)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)
    
    def Kff(self, sigma, l) -> np.ndarray:
        return sigma ** 2 * np.exp(-0.5 / l ** 2 * self.dist)
    
    def K_L_alpha(self, theta):
        if (self.K_cache is not None) and (self.L_cache is not None) and (self.alpha_cache is not None) and np.allclose(theta, self.theta_cache,atol=self.cache_atol,rtol=0):
            return self.K_cache, self.L_cache, self.alpha_cache
        K = self.Kff(theta[0], theta[1]) + self.params["noise"] * np.eye(self.n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.train_y))  # Kinv @ train_y
        self.theta_cache = theta
        self.K_cache = K
        self.L_cache = L
        self.alpha_cache = alpha
        return K, L, alpha
    



