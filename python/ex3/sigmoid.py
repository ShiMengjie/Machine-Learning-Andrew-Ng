import numpy as np
# 计算sigmoid函数值
def sigmoid(z):
	g = 1/(1+np.exp(-z)) 
	
	return g
