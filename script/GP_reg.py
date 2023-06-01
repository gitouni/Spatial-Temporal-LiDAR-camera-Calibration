import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

def get_sklearn_gpr():
    kernel = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
    return gpr

def vectorization(x1:np.ndarray, x2:np.ndarray):
    if(len(x1.shape)==len(x2.shape)==1):
        return (x1[:,None]-x2[None,:])**2
    # x1,x2 : N,d
    dist_matrix:np.ndarray = (x1[:,None,...]-x2[None,...])**2  # (1,N,d) - (N,1,d) -> (N,N,d)
    dist_matrix = np.sum(dist_matrix,axis=-1)  # (N,N,d) -> (N,N)
    return dist_matrix

def rbf_kernel(x1:np.ndarray, x2:np.ndarray, l=1.0, sigma=1.0)->np.ndarray:
    """More efficient approach."""
    dist_matrix = vectorization(x1,x2)
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
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X)
            loss = self.train_y.T.dot(np.linalg.inv(Kyy+1e-8)).dot(self.train_y) + np.linalg.slogdet(Kyy)[1] + self.train_X.shape[0] * np.log(2 * np.pi)
            return loss

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True
        
    def hyperloss(self, param):
        sigma,l = param
        Kyy = rbf_kernel(self.train_X,self.train_X,l,sigma)
        return self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + np.linalg.slogdet(Kyy)[1]
        
    def predict(self, X:np.ndarray):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(Kff.shape[0]))  # (N, N)
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        
        return mu, cov

    def kernel(self, x1, x2):
        return rbf_kernel(x1,x2,self.params['l'],self.params['sigma_f'])
    



