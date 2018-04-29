import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from plotData import plot_data
import costFunction as cf
import plotDecisionBoundary as pdb
from sigmoid import sigmoid
from predict import predict
"""
Logistic Regression 进行分类
"""

# ====== 1.读取数据并绘制散点图 ======
dataset = np.loadtxt('ex2data1.txt',delimiter =',')
X = dataset[:,0:2]
Y = dataset[:,2]

plot_data(X,Y)
plt.legend(['Admitted', 'Not admitted'])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
# plt.show()

# ====== 2.计算代价和梯度 ======
(m,n) = X.shape
X = np.c_[np.ones(m),X]	# 添加偏置维度
# 初始theta值和lambda值
init_theta = np.zeros(X.shape[1])
lmd = 1
# 计算代价和梯度
cost,grad = cf.cost_Function_Reg(X,Y,init_theta,lmd)
print('Cost at initial theta (zeros): {:0.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n{}'.format(grad))
print('Expected gradients (approx): \n-0.1000\n-12.0092\n-11.2628')

# 使用非0的theta测试函数
test_theta = np.array([-24,0.2,0.2])	#生成的是列向量
cost, grad = cf.cost_Function_Reg(X,Y,test_theta,0)
print('Cost at test theta (zeros): {:0.3f}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at test theta: \n{}'.format(grad))
print('Expected gradients (approx): \n0.043\n2.566\n2.647')

# ======= 3.使用优化函数来优化求解 ========
def cost_func(t):
	return cf.cost_Function_Reg(X,Y,t,0)[0]

def grad_func(t):
	return cf.cost_Function_Reg(X,Y,t,0)[1]
# 使用opt.fmin_bfgs()来获得最优解
theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=init_theta, maxiter=400, full_output=True, disp=False)
print('Cost at theta found by fmin: {:0.3f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: \n{}'.format(theta))
print('Expected Theta (approx): \n-25.161\n0.206\n0.201')

# 使用计算出的theta来绘制分界线
pdb.plot_Decision_Boundary(X,Y,theta)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
# plt.show()

# ====== 4.预测和计算准确率 ======
prob = sigmoid(np.array([1, 45, 85]).T.dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of {:0.4f}'.format(prob))
print('Expected value : 0.775 +/- 0.002')

P = predict(X,theta)
print('Train accuracy:{}'.format(np.mean(P == Y)*100))
print('Expected accuracy (approx): 89.0')