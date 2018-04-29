import numpy as np

# 初始化网络参数
def rand_init_weights(L_in,L_out):
	epsilon = np.sqrt(6) / np.sqrt(L_in + L_out)
	init_theta = np.random.random((L_out,L_in+1)) * 2*epsilon - epsilon

	return init_theta
