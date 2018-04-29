import numpy as np

from computeCost import compute_cost
"""梯度下降法"""
def gradient_descent(X,Y,theta_init,alpha,iter_num):
	# 样本个数
	m = Y.shape[0]
	# 代价的历史值
	J_history = np.zeros(iter_num)
	theta = theta_init
	# 进行迭代计算
	for num in range(0,iter_num):
		# 计算每一个theta值下的代价值
		J_history[num] = compute_cost(X,Y,theta)
		# 根据公式计算梯度，来更新theta的值
		hyp = np.dot(X,np.transpose(theta))
		theta = theta - alpha * np.dot(np.transpose(hyp -Y),X) / m

	return theta,J_history

