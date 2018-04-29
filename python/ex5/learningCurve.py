import numpy as np

from linearCostFunction import linear_cost_function
from trainLinearRegression import train_linear_reg

def learning_curve(X,Y,Xval,Yval,lmd):
	m = X.shape[0]
	error_train = np.zeros(m)
	error_val = np.zeros(m)

	for num in range(m):
		theta = train_linear_reg(X[0:num+1,:],Y[0:num+1],lmd)

		error_train[num],_ = linear_cost_function(X[0:num+1,:],Y[0:num+1],theta,lmd)
		error_val[num],_ = linear_cost_function(Xval,Yval,theta,lmd)

	return error_train,error_val