import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

from linearCostFunction import linear_cost_function
from trainLinearRegression import train_linear_reg
from learningCurve import learning_curve
from polyFeatures import ploy_feature
from featuresNormalize import feature_nomalize
from plotFit import plot_fit
from validationCurve import validation_curve
# ============================== 1.读取并显示数据 ==============================
data = scio.loadmat('ex5data1.mat')
X = data['X']
Y = data['y'].flatten()

Xval = data['Xval']
Yval = data['yval'].flatten()

Xtest = data['Xtest']
Ytest = data['ytest'].flatten()

plt.figure(1)
plt.scatter(X,Y,c='r',marker='x')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water folowing out of the dam (y)')
# plt.show()

# ============================ 2.计算代价和梯度 ==============================
(m,n)= X.shape
theta = np.ones((n+1))
lmd=1
cost,grad = linear_cost_function(np.column_stack((np.ones(m),X)),Y,theta,lmd)
print('Cost at theta = [1  1]: {:0.6f}\n(this value should be about 303.993192)'.format(cost))
print('Gradient at theta = [1  1]: {}\n(this value should be about [-15.303016  598.250744]'.format(grad))

# =========================== 3.训练线性回归
lmd = 0
theta = train_linear_reg(np.column_stack((np.ones(m),X)),Y,lmd)
plt.plot(X,np.column_stack((np.ones(m),X)).dot(theta))
# plt.show()

# =========================== 4.线性回归的学习曲线 ==============
lmd = 0
error_train,error_val = learning_curve(np.column_stack((np.ones(m),X)),Y,
						np.column_stack((np.ones(Yval.size),Xval)),Yval,lmd)
plt.figure(2)
plt.plot(range(m),error_train,range(m),error_val)
plt.title('Learning Curve for Linear Regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
# plt.show()

# =============================== 5.投影特征为多项式 ================
p = 8
# 投影和标准化训练集
X_poly = ploy_feature(X,p)
X_poly,mu,sigma = feature_nomalize(X_poly)
X_poly = np.column_stack((np.ones(Y.size),X_poly))

# 投影和标准化验证集
X_poly_val = ploy_feature(Xval,p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.column_stack((np.ones(Yval.size),X_poly_val))

# 投影和标准化测试集
X_poly_test = ploy_feature(Xtest,p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.column_stack((np.ones(Ytest.size),X_poly_test))

print('Normalized Training Example 1 : \n{}'.format(X_poly[0]))

# ======================== 6.多项式特征的学习曲线
lmd = 0
# 绘制拟合曲线
theta = train_linear_reg(X_poly,Y,lmd)
x_fit,y_fit = plot_fit(np.min(X),np.max(X),mu,sigma,theta,p)
plt.figure(3)
plt.scatter(X,Y,c='r',marker='x')
plt.plot(x_fit,y_fit)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water folowing out of the dam (y)')
plt.ylim([0, 60])
plt.title('Polynomial Regression Fit (lambda = {})'.format(lmd))
# plt.show()
# 计算代价误差
error_train, error_val = learning_curve(X_poly, Y, X_poly_val, Yval, lmd)
plt.figure(4)
plt.plot(np.arange(m), error_train, np.arange(m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(lmd))
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
# plt.show()
print('Polynomial Regression (lambda = {})'.format(lmd))
print('# Training Examples\tTrain Error\t\tCross Validation Error')
for i in range(m):
    print('  \t{}\t\t{}\t{}'.format(i, error_train[i], error_val[i]))

# ======================= 7.通过交叉验证集选择正则项系数lambda
lambda_vec,error_train,error_val = validation_curve(X_poly,Y,X_poly_test,Ytest)
plt.figure(5)
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Test Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()