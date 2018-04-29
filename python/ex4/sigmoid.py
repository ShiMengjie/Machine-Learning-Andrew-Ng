import numpy as np


# 计算sigmoid函数值
def sigmoid(z):
	g = 1/(1+np.exp(-z)) 
	
	return g

def sigmoid_gradient(z):
	grad = sigmoid(z) * (1-sigmoid(z))

	return grad
