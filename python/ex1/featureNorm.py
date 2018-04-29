import numpy as np

def feature_normalize(Xdata):
	# 计算每一维度的均值
	X_mean = np.mean(Xdata,axis=0)
	X_std = np.std(Xdata,axis=0)
	X_norm = np.divide(np.subtract(Xdata,X_mean),X_std)

	return X_norm,X_mean,X_std
