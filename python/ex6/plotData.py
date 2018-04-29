import matplotlib.pyplot as plt

# 对不同类别的数据，绘制不同颜色的散点图
def plot_data(X,Y):
    pos = Y == 1
    neg = Y == 0

    plt.figure()
    plt.scatter(X[pos,0],X[pos,1],c='red')
    plt.scatter(X[neg,0],X[neg,1],c='blue')
