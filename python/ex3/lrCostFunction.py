import numpy as np
from sigmoid import sigmoid
# 这个函数是用来求代价和梯度，和前面ex2的costfunction相同
def lr_cost_function(X,Y,theta,lmd):
	m = X.shape[0]
	g = sigmoid(X.dot(theta))
	# g.shape = [m,]
	# Y.shape = [m,1]
	# 两者相减得到的是[m,m]
	Y = Y.reshape(Y.size)

	cost = (-Y.T).dot(np.log(g)) - ((1-Y).T).dot(np.log(1-g))
	cost = cost /(m) + lmd * (theta.T).dot(theta) / (2*m)
	
	grad = (X.T).dot(g-Y)/ m
	grad[0] = grad[0]
	grad[1:] = grad[1:] + (lmd * theta[1:])/m

	return cost,grad