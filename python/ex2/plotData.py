import matplotlib.pyplot as plt

# 用两种不同的颜色，在同一张图像上，绘制两个类别的数据散点图
def plot_data(X,Y):
	plt.figure();
	pos = Y == 1
	neg = Y == 0
	plt.scatter(X[neg,0],X[neg,1],c='black',marker='o',s=20)
	plt.scatter(X[pos,0],X[pos,1],c='red',marker='o',s=20)
