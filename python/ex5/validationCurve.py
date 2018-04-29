import numpy as np
from linearCostFunction import linear_cost_function
from trainLinearRegression import train_linear_reg


def validation_curve(X,Y,Xval,Yval):
	lambda_vec = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
	error_train = np.zeros(lambda_vec.size)
	error_val = np.zeros(lambda_vec.size)

	for num in range(lambda_vec.size):
		lmd = lambda_vec[num]
		theta = train_linear_reg(X,Y,lmd)
		error_train[num],_ = linear_cost_function(X,Y,theta,lmd)
		error_val[num],_ = linear_cost_function(Xval,Yval,theta,lmd)

	return lambda_vec,error_train,error_val