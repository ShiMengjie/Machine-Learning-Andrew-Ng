import numpy as np
from sigmoid import sigmoid
# 计算代价和梯度
def cost_Function_Reg(X,Y,theta,lmd):
	m = X.shape[0]
	Z = np.dot(X,theta)
	g = sigmoid(Z)

	cost = - (Y.T).dot(np.log(g)) - ((1-Y).T).dot(np.log(1-g))
	cost = cost /(m) + lmd * (theta.T).dot(theta) / (2*m)
	# g.shape = [m,]
	# Y.shape = [m,1]
	# 两者相减得到的是[m,m]
	grad = (X.T).dot(g-Y.reshape(Y.size))/ m
	grad[0] = grad[0]
	grad[1:] = grad[1:] + (lmd * theta[1:])/m

	return cost,grad
