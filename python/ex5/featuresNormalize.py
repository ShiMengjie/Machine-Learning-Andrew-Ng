import numpy as np
# 把数据特征标准化
def feature_nomalize(X):
	mu = np.mean(X,0)
	sigma = np.std(X,0,ddof=1)
	X_norm = (X - mu)/sigma

	return X_norm,mu,sigma


