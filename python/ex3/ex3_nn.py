import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt


from displayData import display_data
from predict import predict_nn


# 使用简单的神经网络对“手写数字”进行训练和识别
# ========================= 1.读取数据，显示随机样例 ===========================
data = scio.loadmat('ex3data1.mat')
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

# ======================= 2.读取神经网络的参数 =================================
weight = scio.loadmat('ex3weights.mat')
# theta1.shape=(25,401)，隐藏层有25个节点，输入数据为401维(添加了1个维度的偏置)
# Theta2.shape=(10,26)，10输出层有10个节点，隐藏层添加一个维度后，有25个输出
theta1 = weight['Theta1']
theta2 = weight['Theta2']

P = predict_nn(X,theta1,theta2)
print('Training set accuracy: {}'.format(np.mean(P == Y)*100))

# ======================= 3.随机选取样本，并显示预测结果 =========================
rp = np.random.permutation(range(m))
for i in range(m):
    print('Displaying Example image')
    example = X[rp[i]]
    example = example.reshape((1, example.size))
    display_data(example)

    pred = predict_nn( example,theta1, theta2,)
    print('Neural network prediction: {} (digit {})'.format(pred, np.mod(pred, 10)))
    
    plt.show()

    s = input('Paused - press ENTER to continue, q + ENTER to exit: ')
    if s == 'q':
        break
