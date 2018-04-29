import scipy.optimize as opt
import numpy as np

from sigmoid import sigmoid
from lrCostFunction import lr_cost_function

def one_vs_all(X,Y,num_labels,lmd):
	# 给数据添加偏置维度
	X = np.c_[np.ones(X.shape[0]),X]
	n = X.shape[1]
	# 保存所有theta的集合
	all_theta = np.zeros((num_labels,n))
	# Y中的值是1~10
	for i in range(1,num_labels+1):
		init_theta = np.zeros((n,1));
		y = (Y == i).astype(int)
		
		def cost_func(t):
			return lr_cost_function(X,y,t,lmd)[0]
		def grad_func(t):
			return lr_cost_function(X,y,t,lmd)[1]

		theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=init_theta, maxiter=100, full_output=True, disp=False)
		all_theta[i-1,:] = theta.T

	return all_theta

