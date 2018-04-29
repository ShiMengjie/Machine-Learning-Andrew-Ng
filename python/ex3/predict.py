import numpy as np

from sigmoid import sigmoid

# 根据输入的数据和参数，计算神经网络的输出
def predict_nn(X,theta1,theta2):
	m = X.shape[0]
	X= np.c_[np.ones(m),X]
	# 隐藏层的输入
	z1 = theta1.dot(X.T)
	# 增加一行维度
	z1 = np.row_stack((np.ones(z1.shape[1]),z1))
	# 隐藏层的输出
	A1 = sigmoid(z1)

	# 输出层的输入
	z2 = theta2.dot(A1)
	# 输出层的输出
	A2 = sigmoid(z2.T)

	# 进行预测
	P = np.zeros(m)
	for num in range(m):
		# 找到第num行中，与该行最大值相等的列的下标，此时下标的范围是[0,9]
		# label的范围是[1,10]，需要把下标的值+1
		# np.where()返回的是一个长度为2的元祖，保存的是满足条件的下标
		# 元组中第一个元素保存的是行下标，第二元素保存的是列下标
		index = np.where(A2[num,:] == np.max(A2[num,:]))
		P[num] = index[0][0].astype(int) + 1

	return P

# 向量化logistic的预测函数
def predict_one_vs_all(X,all_theta):
	m = X.shape[0]
	X = np.c_[np.ones(m),X]
	# 标签数10
	num_labels = all_theta.shape[0]
	
	# preds[m][k]是第m个样本属于k的概率
	preds = sigmoid(X.dot(all_theta.T))
	P = np.zeros(m)

	for num in range(m):
		# 找到第num行中，与该行最大值相等的列的下标，此时下标的范围是[0,9]
		# label的范围是[1,10]，需要把下标的值+1
		# np.where()返回的是一个长度为2的元祖，保存的是满足条件的下标
		# 元组中第一个元素保存的是行下标，第二元素保存的是列下标
		index = np.where(preds[num,:] == np.max(preds[num,:]))
		P[num] = index[0][0].astype(int) + 1

	return P