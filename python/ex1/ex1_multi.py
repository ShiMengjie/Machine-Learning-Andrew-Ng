import matplotlib.pyplot as plt
import numpy as np

from gradientDescent import gradient_descent as gd
from featureNorm import feature_normalize
# ======== 1.读取数据并标准化数据的特征 ========
data = np.loadtxt(r'data\ex1data2.txt',delimiter =',')
Xdata = data[:,0:2]
Ydata = data[:,2]
# 对输入数据特征进行标准化
X,mu,sigma = feature_normalize(Xdata)
# print('Mu is:',mu)
# print('Sigma is:',sigma)
X = np.c_[np.ones(X.shape[0]),X]
Y = Ydata

# ======== 2.使用梯度下降法求解 ========
theta_init = np.zeros(X.shape[1])
alpha = 0.05
num_iters = 300

theta,J_history = gd(X,Y,theta_init,alpha,num_iters)
plt.figure()
plt.plot(np.arange(J_history.size),J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
#plt.axis([0,num_iters,0,100])
plt.show()
# theta是数组格式，格式化输出时，不能像下面那样{:0.3f}
print('Theta computed from gradient descent : \n{}'.format(theta))

# ======== 3.预测 ========
Xtest = np.array([1,1650,3])
price = np.dot(Xtest,np.transpose(theta))
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : {:0.3f}'.format(price))