"""计算线性回归的代价和梯度"""
import numpy as np

def linear_cost_function(X,Y,theta,lmd):
	m = X.shape[0]
	hyp = X.dot(theta) - Y
	grad = np.zeros(theta.shape) 

	cost = ((hyp.T).dot(hyp) + lmd * (theta.T).dot(theta))/(2*m)

	temp = (X.T).dot(hyp)
	grad[0] = temp[0]/ m
	grad[1:] = (temp[1:] + lmd * theta[1:])/m
	# grad = ((X.T).dot(hyp) + lmd * np.row_stack((np.zeros(1),)).flatten()) / m
	
	return cost,grad