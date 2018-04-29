import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# 需要自己写的模块
from displayData import display_data
from lrCostFunction import lr_cost_function
from oneVsAll import one_vs_all
from predict import predict_one_vs_all
# 使用Ligistic Regression进行多分类

# ============================== 1.读取数据和初始化 ============================
# 使用scipy.io中的函数读取mat文件，data的格式是字典
data = scio.loadmat('ex3data1.mat')
# 根据关键字，分别获得输入数据和输出的真值
# print(type(Y),type(X)) # X和Y都是numpy.narray格式，也就是数组格式
X = data['X']
Y = data['y']
print(X.shape)

# 随机取出其中的100个样本，显示结果
m = X.shape[0]
# 从[0,m-1]之间，随机生成一个序列
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[1:100],:]
# 显示手写数字样例
display_data(selected)

# ========================= 2.向量化Logistic Rgression =========================
# 测试函数lr_cost_function的功能
"""
theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape((3, 5)).T/10]
y_t = np.array([1, 0, 1, 0, 1])
lmda_t = 3
cost,grad = lr_cost_function(X_t,y_t,theta_t,lmda_t)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
print('Cost: {:0.7f}'.format(cost))
print('Expected cost: 3.734819')
print('Gradients:\n{}'.format(grad))
print('Expected gradients:\n[ 0.146561 -0.548558 0.724722 1.398003]')
"""
# 训练模型

lmd = 0.01
num_labels = 10
all_theta = one_vs_all(X,Y,num_labels,lmd)

# =============================== 3.预测 =======================================
pred = predict_one_vs_all(X,all_theta)
# 这里一定要把Y.shape变成(m,)，否则Y.shape = (m,1)，带入是无效的
Y = Y.reshape(Y.size)
print('Training set accurayc:{}'.format(np.mean(pred == Y)*100))
