import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

from plotData import plot_data
from computeCost import compute_cost
from gradientDescent import *

# ======== 1.载入数据和绘制散点图 ========
print('读取数据，并绘制散点图...\n')
filepath = r'data\ex1data1.txt'
# 从文件中读取数据，读取第0列和第1列，要求文件中每一行的列数相等
dataset = np.loadtxt(filepath, 
	delimiter=',',
	usecols=(0,1))
Xdata = dataset[:,0]
Ydata = dataset[:,1]

# ======== 2.计算代价和梯度 ========
print('进行梯度计算...\n')
# 按照第二维度，把两个数组连接起来
# 给输入数据增加一个偏置维度
X = np.c_[np.ones(Xdata.shape[0]),Xdata]
Y = Ydata
# 初始化参数：theta,iter_num,alpha
theta_init = np.zeros(X.shape[1])
iter_num = 1500
alpha = 0.01
# 计算初始代价
print('Initial cost:',
	str(compute_cost(X,Y,theta_init)),
	'\nThis value should be 32.07')
# 使用梯度下降法进行优化求解
theta_fin,J_history = gradient_descent(X,Y,theta_init,alpha,iter_num)
print('Theta found by gradient descent:',str(theta_fin.reshape(2)))
# 绘制数据散点图和线性回归曲线
plt.figure(0)
plt.scatter(Xdata,Ydata,c='red',marker='o',s=20)
plt.plot(X[:,1],np.dot(X,theta_fin),'b-',lw=3)
plt.xlabel('Population of City in 10,000s',fontsize=10)
plt.ylabel('Profit of City in $10,000',fontsize=10)
plt.legend(['Data Point','Linear Regression'])
plt.show()

# 预测未知数据
Xtest1 = [1,3.5]
Xtest2 = [1,7]
Xtest = np.array([Xtest1,Xtest2])
predict = np.dot(Xtest,theta_fin)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict[0]*10000))
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict[1]*10000))

# ======== 3.可视化代价J(theta0,theta1) ========
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)
# 这个函数功能和matlab中的函数功能相同，生成网格坐标
# xs的每一行都是theta0_vals的复制
# ys的每一列都是theta1_vals的复制
xs,ys = np.meshgrid(theta0_vals,theta1_vals)
J_vals = np.zeros(xs.shape)

for i in range(0,theta0_vals.size):
	for j in range(0,theta1_vals.size):
		t = np.array([theta0_vals[i],theta1_vals[j]])
		J_vals[i][j] = compute_cost(X,Y,t)
J_vals = np.transpose(J_vals)

# 绘制3D曲面图
figure = plt.figure(1)	# 创建一个图像
# 把图像指定为3D视图
ax = Axes3D(figure)
# 绘制3D图像
ax.plot_surface(xs,ys,J_vals,cmap='rainbow')
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()

# 绘制代价的等高椭圆线
plt.figure(2)
# logspace()的作用与matlab中的相同
lvls = np.logspace(-2, 3, 20)
# 指定坐标和值，指定等高线的值，指定色彩
plt.contour(xs, ys, J_vals, levels=lvls, norm = LogNorm())
plt.plot(theta_fin[0], theta_fin[1], 'ro',markersize =6)
plt.show()