import matplotlib.pyplot as plt
import numpy as np
from polyFeatures import ploy_feature
# 输入横坐标范围，计算出拟合曲线在每一点的取值
def plot_fit(min_x,max_x,mu,sigma,theta,p):
	x = np.arange(min_x-15,max_x+25,0.05)

	X_poly = ploy_feature(x,p)
	X_poly -= mu
	X_poly /= sigma
	X_poly = np.column_stack((np.ones(x.size),X_poly))

	Y_fit = X_poly.dot(theta)

	return x,Y_fit

