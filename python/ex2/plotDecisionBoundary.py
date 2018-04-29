import matplotlib.pyplot as plt
import numpy as np
from plotData import plot_data
from mapFeature import map_feature

def plot_Decision_Boundary(X,Y,theta):
	plot_data(X[:,1:3],Y)

	if X.shape[1] <= 3:
		plot_x = np.array([np.min(X[:,1])-2,np.max(X[:,1])+2])
		plot_y = (-1/theta[2]) * (theta[1]*plot_x + theta[0])

		plt.plot(plot_x,plot_y)
		plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'])
		plt.axis([30,100,30,100])
	else:
		u = np.linspace(-1,1.5,50)
		v = np.linspace(-1,1.5,50)
		z = np.zeros((u.size,v.size))

		for i in range(0,u.size):
			for j in range(0,v.size):
				z[i,j] = np.dot(map_feature(u[i],v[j],6),theta,)

		z = z.T # 转置
		cs = plt.contour(u,v,z,level=[0],colors='b',label='Decision Boundary')
		plt.legend([cs.collections[0]], ['Decision Boundary']) # [cs.collections[0]] 显示里面的分界线，与后面列表中的字符串对应
