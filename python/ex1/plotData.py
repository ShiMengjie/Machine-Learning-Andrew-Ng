import matplotlib.pyplot as plt


"""绘制二维散点图"""
def plot_data(X,Y,title,xlabel,ylabel):
	plt.plot(X,Y,'ro',markersize=6)
	plt.title(title,fontsize=20)
	plt.xlabel(xlabel,fontsize=10)
	plt.ylabel(ylabel,fontsize=10)
	plt.ioff()