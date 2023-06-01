#-*-coding:utf-8-*-

import math
import os
from collections.abc import Iterable
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle as pkl
from scipy.optimize import minimize
from scipy.optimize import Bounds
import warnings

def test_f(X):
    return np.sqrt(X[:,0]**2 + 0.5*X[:,1]**2 + 0.5**X[:,2]**2)

def test_constrains():
    con1 = lambda X: 4-(X[:,0]+0.5*X[:,1]+0.5*X[:,2])
    con2 = lambda X: 3-(X[:,0]-0.5*X[:,1]-0.5*X[:,2])
    return dict(neq=[con1,con2])

def test_pso():
    n_size=1000
    x0 = np.array([0,0,0])
    f = lambda x: np.sum((x-np.repeat(x0[None,:],repeats=x.shape[0],axis=0))**2,axis=1)  #  (n_size,) (x-x0)^2
    eq1 = lambda x: np.sum(x,axis=1)-3  # x1 + x2 + x3 = 3
    neq1 = lambda x: 3 - (x[:,0]**2 + x[:,1]**2)  #  3 - (x1^2 + x2^2) <= 0
    neq2 = lambda x: 1 - x[:,2]**2   # 1 - x3^2 <= 0
    lb=-10*np.ones(3)
    ub=10*np.ones(3)
    pso = PSOCO(particle_size=n_size, max_iter=200,
                 sol_size=3,
                 Xmax=ub,
                 Xmin=lb,
                 Vmax=0.1*ub,
                 Vmin=0.1*lb,
                 fitness=f,
                 eq_cons=[eq1],
                 neq_cons=[neq1,neq2]
                 )
    pso.init_Population()
    sol,fval = pso.solve()
    print("Pg:{},fval:{}".format(sol, fval))


class scalar_func():
    def __init__(self,func,inv=False):
        self.func = func
        self.inv = inv
    def __call__(self,x):
        if len(x.shape)==1:
            x = x[None,:]  # extend dim
        y = self.func(x)
        if self.inv:
            y = -y
        return y.item()

class PSO_GUI():
    def __init__(self,
        select_id=[0,1],
        Pmin=[-10,-10],
        Pmax=[10,10],
        grid=30):
        self.id = np.array(select_id)
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.P = []
        self.V = []
        self.Pg = []
        self.fval = []
        self.iter = []
        self.var = []
        self.grid = grid

    def add_item(self,X,V,Pg,fval,iter):
        num = X.shape[1]
        assert num >= 2, "dimension of data ({}) must not smaller than 2.".format(num)
        self.P.append(X[:,self.id])
        self.V.append(V[:,self.id])
        self.Pg.append(Pg)
        self.fval.append(fval)
        self.iter.append(iter)
        self.var.append(np.sqrt(np.sum(np.std(X,axis=0)**2)))

    def plot_all(self,fig_num=6,ncols=3,vol=0.5):
        iter_num = len(self.iter)  # 保存内存数目
        assert fig_num<iter_num, \
            "fig num({}) must be smaller than memory({})".format(fig_num,iter_num)
        plt.figure(1,figsize=[6.4,4.8],dpi=100)
        ax1 = plt.subplot(111)
        ax2 = ax1.twinx()
        line1, = ax1.plot(self.iter,self.fval,'b-',label='fit')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('fitness')
        line2, = ax2.plot(self.iter,self.var,'r--',label='std')
        ax2.set_ylabel('variance')
        plt.legend([line1,line2],['fit','std'])
        plt.title('fval(last):{:.6f}  std(last):{:.6f}'.format(self.fval[-1],self.var[-1]))
        plt.savefig('gpso_fitness.png')

        fig = plt.figure(2,figsize=[10,8],dpi=200)
        nrows = (fig_num + ncols - 1)//ncols
        print("rows:{}  cols:{}".format(nrows,ncols))
        fig.subplots(nrows=nrows,ncols=ncols)
        index = np.zeros(fig_num,dtype=np.int32)
        index[1:] = np.linspace(1,iter_num-1,fig_num-1).astype(np.int32)
        for fig_i,ax in enumerate(fig.axes):
            data_i = index[fig_i]
            drawx = self.P[data_i][:,0]  # (N,2)->(N,),(N,)
            drawy = self.P[data_i][:,1]  # (N,2)->(N,),(N,)
            ax.set_title('iteration {}'.format(self.iter[data_i]))
            ax.plot(drawx,drawy,'b*',markersize=vol,label='partial')
            ax.plot(self.Pg[data_i][0],self.Pg[data_i][1],'r^',markersize=4,label='best')
            ax.set_xlim(self.Pmin[0],self.Pmax[0])
            ax.set_ylim(self.Pmin[1],self.Pmax[1])
        lines, labels = ax.get_legend_handles_labels()
        fig.legend(lines,labels,loc='upper right')
        plt.suptitle('distribution of partials by iteration')
        plt.savefig('gpso_partials.png')
        try:
            fig = plt.figure(3,figsize=[16,8],dpi=200)
            fig.subplots(nrows=nrows,ncols=ncols)
            regionx = np.hstack((np.array([self.Pmin[0]]),np.linspace(self.Pmin[0],self.Pmax[0],self.grid-1))).astype(np.int32)
            regiony = np.hstack((np.array([self.Pmin[1]]),np.linspace(self.Pmin[1],self.Pmax[1],self.grid-1))).astype(np.int32)
            Y, X = np.mgrid[self.Pmin[0]:self.Pmax[0]:complex(0,self.grid),self.Pmin[1]:self.Pmax[1]:complex(0,self.grid)]
            U, V = np.zeros_like(X), np.zeros_like(Y)
            for fig_i,ax in enumerate(fig.axes):
                data_i = index[fig_i]
                drawx = self.P[data_i][:,0]  # (N,2)->(N,),(N,)
                drawy = self.P[data_i][:,1]  # (N,2)->(N,),(N,)
                ix = np.digitize(drawx,regionx)-1
                iy = np.digitize(drawy,regiony)-1
                U[ix,iy] = self.V[data_i][:,0]
                V[ix,iy] = self.V[data_i][:,1]
                intense = np.sqrt(U**2+V**2)
                strm = ax.streamplot(X,Y,U,V,color=intense,linewidth=1,cmap='autumn')
                ax.plot(self.Pg[data_i][0],self.Pg[data_i][1],'b^',markersize=4,label='best')
                ax.set_title('iteration {}'.format(self.iter[data_i]))
                ax.set_xlim(self.Pmin[0],self.Pmax[0])
                ax.set_ylim(self.Pmin[1],self.Pmax[1])
            fig.colorbar(strm.lines,ax=fig.axes)
            p = mlines.Line2D([], [], linewidth=0, color='blue', marker='^',
                            markersize=4, label='best')
            fig.legend(handles=[p],loc='lower center')
            plt.suptitle("stream of partials")
            plt.savefig('gpso_stream.png')
        except Exception as e:
            warnings.warn("stream figure failed to generate! Exception:{}".format(e),RuntimeWarning)
    




class PSOCO():
    def __init__(self,
        particle_size=2000,
        max_iter=1000,
        sol_size=3,
        fitness=None,
        eq_cons=[],
        neq_cons=[],
        Xmin=[-10,-10,-10],
        Xmax=[10,10,10],
        Vmin=[-4,-4,-4],
        Vmax=[4,4,4],
        record_T=5,
        min_loss = None,
        hybrid=True):
        '''
        Particle Swarm Optimization Constraint Optimization
        Args:
            particle_size (int): 粒子数量
            max_iter (int): 最大迭代次数
            sol_size (int): 解的维度
            fitness (callable function): fitness函数, 接受参数 x 为解
            neq_cons (list): 不等式约束条件，全部表示为 <= 0的形式
        '''
        self.c1 = 2 
        self.c2 = 2 
        self.w = 1.2 # 逐渐减少到0.1 
        phi = self.c1 + self.c2 
        self.kai = 2/abs(2-phi-math.sqrt(phi**2-4*phi))  # http://eprints.gla.ac.uk/44801/1/44801.pdf
        self.xmax = Xmax
        self.xmin = Xmin
        self.vmax = Vmax # 最大速度，防止爆炸
        self.vmin = Vmin
        self.particle_size = particle_size 
        self.max_iter = max_iter
        self.sol_size = sol_size
        self.record_T = record_T
        # pso parameters 
        self.X = np.zeros((self.particle_size, self.sol_size))
        self.V = np.zeros((self.particle_size, self.sol_size))
        self.pbest = np.zeros((self.particle_size, self.sol_size))   #个体经历的最佳位置和全局最佳位置  
        self.gbest = np.zeros((1, self.sol_size))  
        self.p_fit = np.zeros(self.particle_size) # 每个particle的最优值
        self.fit = float("inf")
        self.iter = 1
        self.neq_cons = neq_cons
        self.eq_cons = eq_cons
        self.sub_fitness = fitness
        self.hybrid = hybrid
        self.gui = PSO_GUI(Pmin=self.xmin,Pmax=self.xmax)
        self.non_con = False
        self.min_loss = min_loss
        self.check()

    def check(self):
        assert len(self.xmax) == self.sol_size,"size of xmax({}) is incompatiable with sol_size({})".format(len(self.xmax),self.sol_size)
        assert len(self.xmin) == self.sol_size,"size of xmin({}) is incompatiable with sol_size({})".format(len(self.xmin),self.sol_size)
        assert len(self.vmax) == self.sol_size,"size of vmax({}) is incompatiable with sol_size({})".format(len(self.vmax),self.sol_size)
        assert len(self.vmin) == self.sol_size,"size of vmin({}) is incompatiable with sol_size({})".format(len(self.vmin),self.sol_size)
        self.xmax = np.array(self.xmax)
        self.xmin = np.array(self.xmin)
        self.vmax = np.array(self.vmax)
        self.vmin = np.array(self.vmin)

        assert isinstance(self.neq_cons,Iterable), "neq_cons must be iterable (can be empyt tuple)"
        for cons in self.neq_cons:
            if not callable(cons):
                raise Exception("neq_constrain is not callable!")
        assert isinstance(self.eq_cons,Iterable), "neq_cons must be iterable (can be empyt tuple)"
        for cons in self.eq_cons:
            if not callable(cons):
                raise Exception("eq_constrain is not callable!")
        if not callable(self.fitness):
            raise Exception("Fitness is not callable!")
        print('PSO parameters are valid!')
        if len(self.neq_cons) == 0 and len(self.eq_cons) == 0:
            self.non_con = True
        
    def fitness(self, x, k):
        '''fitness函数 + 惩罚项'''
        obj = self.sub_fitness(x)
        if self.non_con:
            return obj
        else:
            return obj + self.h(k) * self.H(x)
    
    def init_Population(self, X0=None):  
        '''初始化粒子'''
        self.X = np.random.uniform(size=(self.particle_size, self.sol_size),low=self.xmin,high=self.xmax)
        self.V = np.random.uniform(size=(self.particle_size, self.sol_size),low=self.vmin,high=self.vmax)
        if X0 is not None:
            assert isinstance(X0, np.ndarray)
            if len(X0.shape) == 1:
                self.X[0, :] = X0
            else:
                self.X[:X0.shape[0], :] = X0
        self.pbest = self.X   # 每个粒子的历史最优值
        print("initialize Population...")
        self.p_fit = self.fitness(self.X, 1)   # 当前粒子的fitness
        best_idx = np.argmin(self.p_fit) # 当前粒子群最优fitness的粒子序号
        best = self.p_fit[best_idx]  # 当前粒子群最优值
        self.fit = best  # 更新最优fitness
        self.gbest = self.X[best_idx]   # 更新最优粒子的位置
    
    def solve(self):  
        '''求解'''
        w_step = (self.w - 0.1) / self.max_iter
        t = tqdm(range(self.max_iter))
        with t:
            for k in range(1, self.max_iter+1):
                tmp_obj = self.fitness(self.X, k)  # last best fitness
                # 更新pbest 
                stack = np.hstack((tmp_obj.reshape((-1, 1)), self.p_fit.reshape((-1, 1))))
                best_arg = np.argmin(stack, axis=1).ravel().tolist()
                self.p_fit = np.minimum(tmp_obj.flatten(), self.p_fit.flatten())  # this best fitness
                X_expand = np.expand_dims(self.X, axis=2)  # 粒子的扩展  (N,dim,2)
                p_best_expand = np.expand_dims(self.pbest, axis=2)  # 每个粒子的历史最优值的扩展  (N,dim,2)
                concat = np.concatenate((X_expand, p_best_expand), axis=2)
                self.pbest = concat[range(0, len(best_arg)), :, best_arg]  # [200,2,2] -> [200,2]

                # 更新fit和gbest 
                best_idx = np.argmin(self.p_fit)
                best = self.p_fit[best_idx]
                if best < self.fit:
                    self.fit = best 
                    self.gbest = self.X[best_idx]

                # 更新速度 

                # 分粒子更新
                # for i in range(self.particle_size):  
                #     self.V[i] = self.w*self.V[i] + self.c1*random.random()*(self.pbest[i] - self.X[i]) + \
                #                 self.c2*random.random()*(self.gbest - self.X[i])  
                #     self.X[i] = self.X[i] + self.V[i] 

                rand1 = np.random.random(size=(self.particle_size, self.sol_size))
                rand2 = np.random.random(size=(self.particle_size, self.sol_size))
                # update V with 3 factors
                self.V = self.kai * (self.w*self.V + self.c1*rand1*(self.pbest - self.X) + \
                            self.c2*rand2*(self.gbest - self.X))
                np.clip(self.V,self.vmin,self.vmax,out=self.V)  # clip with vmin and vmax
                self.X = self.X + self.V  
                np.clip(self.X,self.xmin,self.xmax,out=self.X)
                self.w -= w_step
                if self.record_T > 0:
                    if k%self.record_T == 0 or k==1:
                        self.gui.add_item(self.X,self.V,self.gbest,self.fit,k)
                    t.set_description_str("Iter {:03d}|{:03d}".format(k,self.max_iter))
                t.set_postfix(iter_best=best,gbest=self.fit)
                t.update(1)
                if self.min_loss is not None and self.fit <= self.min_loss:
                    print('acheived min_loss {} in Iteration {}'.format(self.min_loss,k))
                    break
                
        self.p_fit = self.fitness(self.X,self.max_iter)
        best_idx = np.argmin(self.p_fit)
        best = self.p_fit[best_idx]
        if best < self.fit:
            self.fit = best 
            self.gbest = self.X[best_idx]
        sol = self.gbest
        fval = best
        if self.hybrid:    
            print('first opt:\nsol:{}\nfval:{}'.format(sol,fval))
            for i,eq_con in enumerate(self.eq_cons):
                print("eq constrain {}:{}".format(i+1,eq_con(sol[None,:])))
            for i,neq_con in enumerate(self.neq_cons):
                print("ineq constrain {}:{}".format(i+1,neq_con(sol[None,:])))
            cons = []
            for neq_con in self.neq_cons:
                neq_scon = scalar_func(neq_con,inv=True)
                cons.append(dict(type='ineq',fun=neq_scon))
            for eq_con in self.eq_cons:
                eq_scon = scalar_func(eq_con)
                cons.append(dict(type='eq',fun=eq_scon))
            obj_func = scalar_func(self.sub_fitness)
            bnds = Bounds(lb=self.xmin.tolist(),ub=self.xmax.tolist())

            print("Using 'SLSQP' for refinement:")
            res = minimize(obj_func,sol,method='SLSQP',
                        constraints=cons,bounds=bnds,options=dict(disp=True,ftol=1e-8,eps=1e-8)
                        )
            sol = res.x
            fval = res.fun
            print('second opt:\nfval:{}'.format(fval))
            for i,eq_con in enumerate(self.eq_cons):
                print("eq constrain {}:{}".format(i+1,eq_con(sol[None,:])))
            for i,neq_con in enumerate(self.neq_cons):
                print("ineq constrain {}:{}".format(i+1,neq_con(sol[None,:])))
        os.makedirs("results",exist_ok=True)
        with open('results/gpso_gui.pth','wb')as f:
            pkl.dump(self.gui,f)
        return sol,fval 
    
    # relative violated function
    def q(self, g):
        return np.maximum(0, g)
    
    # power of penalty function 
    def gamma(self, qscore):
        result = np.zeros_like(qscore)
        result[qscore >= 1] = 2
        result[qscore < 1] = 1 
        return result
    
    # multi-assignment function
    def theta(self, qscore):
        result = np.zeros_like(qscore)
        result[qscore < 0.001] = 10 
        result[qscore <= 0.1] = 20 
        result[qscore <= 1] = 100
        result[qscore > 1] = 300
        return result
    
    # penalty score 
    def h(self, k):
        return k * math.sqrt(k)
    
    # penalty factor
    def H(self, x):
        if len(self.neq_cons)>0:
            neq_num = len(self.neq_cons)
            neq_score = [self.q(self.neq_cons[i](x)) for i in range(neq_num)]
            neq_reses = [self.theta(neq_score[i]) * np.power(neq_score[i], self.gamma(neq_score[i])) for i in range(neq_num)]
            neq_res = sum(neq_reses)
        else:
            neq_res = 0
        if len(self.eq_cons)>0:
            eq_num = len(self.eq_cons)
            eq_score = [self.eq_cons[i](x)**2 for i in range(eq_num)]
            eq_reses = [self.theta(eq_score[i]) * eq_score[i] for i in range(eq_num)]
            eq_res = sum(eq_reses)
        else:
            eq_res = 0
        return neq_res + eq_res

if __name__=="__main__":
    test_pso()
    # pso_gui = pkl.load(open('results/gpso_gui.pth','rb'))
    # pso_gui.grid=30
    # pso_gui.plot_all()
