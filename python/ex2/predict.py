import numpy as np
from sigmoid import sigmoid

def predict(X,theta):
	P = sigmoid(X.dot(theta))

	P[P>=0.5] =1
	P[P<0.5] =0

	return P
