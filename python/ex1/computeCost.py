"""计算代价值"""
import numpy as np
def compute_cost(X,Y,theta):
	# 两个数组作矩阵乘积
	# 当两个数组的维度不能直接进行矩阵乘法时，dot会把后面的参数进行转置
	hypthesis = np.dot(X,np.transpose(theta))
	# 先转置再做矩阵乘法
	cost = np.dot(np.transpose(hypthesis - Y),(hypthesis -Y))
	cost = cost / (2 * X.shape[0])
	return cost