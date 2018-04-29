import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from displayData import display_data
from costFunction import nn_cost_function
from sigmoid import sigmoid_gradient
from randInitializeWeights import rand_init_weights
from checkNNGradients import check_nn_gradients
from predict import predict_nn
# ==================== 1.读取数据，并显示随机样例 ==============================
# 使用scipy.io中的函数读取mat文件，data的格式是字典
data = scio.loadmat('ex4data1.mat')
# 根据关键字，分别获得输入数据和输出的真值
# print(type(Y),type(X)) # X和Y都是numpy.narray格式，也就是数组格式
X = data['X']
Y = data['y'].flatten()

# 随机取出其中的100个样本，显示结果
m = X.shape[0]
# 从[0,m-1]之间，随机生成一个序列
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[1:100],:]
# 显示手写数字样例
display_data(selected)
# plt.show()

# ==================== 2.读取参数，并计算代价 ==================================
weights = scio.loadmat('ex4weights.mat')
theta1 = weights['Theta1']	# 25*401
theta2 = weights['Theta2']	# 10*26
# theta1.flatten()把数组变成一列的形式，等价于theta1.reshape(theta1.size)
# 把两个列向量按行拼接起来，此时nn_paramters.shape=(10285,)
nn_paramters = np.concatenate([theta1.flatten(),theta2.flatten()],axis =0)
# 设置参数
input_layer = 400
hidden_layer = 25
out_layer = 10
# 计算代价
lmd = 0
cost,grad = nn_cost_function(X,Y,nn_paramters,input_layer,hidden_layer,out_layer,lmd)
print('Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.287629)'.format(cost))
# 带入正则项
lmd = 1
cost,grad = nn_cost_function(X,Y,nn_paramters,input_layer,hidden_layer,out_layer,lmd)
print('Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.383770)'.format(cost))
# 验证sigmoid的梯度
g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1  -0.5  0  0.5  1]:\n{}'.format(g))

# =========================== 3.初始化网络参数 =================================
random_theta1 = rand_init_weights(input_layer,hidden_layer)
random_theta2 = rand_init_weights(hidden_layer,out_layer)
rand_nn_parameters = np.concatenate([random_theta1.flatten(),random_theta2.flatten()])
# 检查BP算法
lmd =3
check_nn_gradients(lmd)
debug_cost, _ = nn_cost_function(X,Y,nn_paramters,input_layer,hidden_layer,out_layer,lmd)
print('Cost at (fixed) debugging parameters (w/ lambda = {}): {:0.6f}\n(for lambda = 3, this value should be about 0.576051)'.format(lmd, debug_cost))

# ========================== 4.训练NN ==========================================
lmd = 1
def cost_func(p):
    return nn_cost_function(X,Y,p,input_layer,hidden_layer,out_layer,lmd)[0]

def grad_func(p):
    return nn_cost_function(X,Y,p,input_layer,hidden_layer,out_layer,lmd)[1]

nn_params, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=rand_nn_parameters, maxiter=400, disp=True, full_output=True)

# Obtain theta1 and theta2 back from nn_params
theta1 = nn_params[:hidden_layer * (input_layer + 1)].reshape(hidden_layer, input_layer + 1)
theta2 = nn_params[hidden_layer * (input_layer + 1):].reshape(out_layer, hidden_layer + 1)

# ======================= 5.可视化系数和预测 ===================================
display_data(theta1[:, 1:])
plt.show()

pred = predict_nn(X,theta1, theta2)
print('Training set accuracy: {}'.format(np.mean(pred == Y)*100))
