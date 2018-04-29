import numpy as np

from sigmoid import sigmoid
from sigmoid import sigmoid_gradient

def nn_cost_function(X,Y,nn_paramters,input_layer,hidden_layer,out_layer,lmd=0):
	# 取出theta1和theta2
	theta1 = nn_paramters[:hidden_layer*(input_layer+1)].reshape(hidden_layer,input_layer+1)
	theta2 = nn_paramters[hidden_layer*(input_layer+1):].reshape(out_layer,hidden_layer+1)
	# 获取样本数目
	m = Y.size

	# 输入层的输出等于输入，X增加一列偏置维度
	a1 = np.column_stack((np.ones(X.shape[0]),X))	# 5000*401
	# 隐藏层的输入和输出
	z2 = a1.dot(theta1.T)	# 5000*25
	a2 = sigmoid(z2)
	a2 = np.column_stack((np.ones(a2.shape[0]),a2))	# 5000*26
	# 输出层的输入和输出
	z3 = a2.dot(theta2.T) # 5000*10
	a3 = sigmoid(z3)	# 5000*10
	# a3[m,k]表示第m个样本，预测属于k的概率(因为激活函数是logistic函数)
	# 根据Y的值，也转换成和a3相同格式的数组
	# yk中每一行只能有一列值为1，yk[m,k]=1表示第m个样本的真实输出是k，其他列为0
	yk = np.zeros((m,out_layer))
	# Y中的取值范围是[1,10]，而yk中的列下标范围是[0,9]，需要注意
	for num in range(Y.size):
		yk[num,Y[num]-1] = 1
	# 计算代价，因为输出层的激活函数是logistic函数，所有代价也是以logistic regression代价函数
	cost_arr = - yk * np.log(a3) - (1-yk) * np.log(1-a3)
	cost = cost_arr.sum()/m + lmd /(2*m) *( (theta1[:,1:] **2).sum() + (theta2[:,1:] **2).sum())

	# 使用BP算法计算梯度
	delta3 = a3 - yk # 5000*10

	delta2 = delta3.dot(theta2) * sigmoid_gradient(np.column_stack((np.ones(z2.shape[0]),z2)))
	delta2 = delta2[:,1:]	# 5000*25
	# theta1的梯度
	theta1_grad = np.zeros(theta1.shape)  # 25 x 401
	theta1_grad = theta1_grad + (delta2.T).dot(a1)	# 25*401
	nn_parameter1_grad = theta1_grad/m + (lmd/m) * np.column_stack((np.zeros(theta1.shape[0]),theta1[:,1:]))
	# theta2的梯度
	theta2_grad = np.zeros(theta2.shape)  # 10 x 26
	theta2_grad = theta2_grad + (delta3.T).dot(a2)	# 10*26
	nn_parameter2_grad = theta2_grad/m + (1/m) * np.column_stack((np.zeros(theta2.shape[0]),theta2[:,1:]))
	# 返回梯度
	grad = np.concatenate([nn_parameter1_grad.flatten(),nn_parameter2_grad.flatten()])

	return cost,grad