import numpy as np

def ploy_feature(X,p):
	m = X.shape[0]
	X_poly = np.zeros((m,p))

	for num in range(1,p+1):
		X_poly[:,num-1] = X.flatten() ** num

	return X_poly
