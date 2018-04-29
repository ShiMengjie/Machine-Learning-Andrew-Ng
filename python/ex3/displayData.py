import matplotlib.pyplot as plt
import numpy as np

# 函数说明：把输入的图像数据X进行重新排列，显示在一个面板figurePane中，
# 面板中有多个小imge用来显示每一行数据

def display_data(x):
	(m,n) = x.shape

	# 设置每个小图例的宽度和高度
	width = np.round(np.sqrt(n)).astype(int)
	height = (n / width).astype(int)

	# 设置图片的行数和列数
	rows = np.floor(np.sqrt(m)).astype(int)
	cols = np.ceil(m / rows).astype(int)

	# 设置图例之间的间隔
	pad = 1

	# 初始化图像数据
	display_array = -np.ones((pad + rows*(height+pad),
							  pad + cols*(width + pad)))

	# 把数据按行和列复制进图像中
	current_image = 0
	for j in range(rows):
		for i in range(cols):
			if current_image > m:
				break
			# [:,np.newaxis]可以让指定的那一列数据以列的形式返回和指定
			# 否则返回的是行的形式
			max_val = np.max(np.abs(x[current_image,:]))
			display_array[pad + j*(height + pad) + np.arange(height),
						  pad + i*(width + pad) + np.arange(width)[:,np.newaxis]] = \
						  x[current_image,:].reshape((height,width)) / max_val
			current_image += 1
		if current_image > m :
			break

	# 显示图像
	plt.figure()
	# 设置图像色彩为灰度值，指定图像坐标范围
	plt.imshow(display_array,cmap = 'gray',extent =[-1,1,-1,1])
	plt.axis('off')
	plt.title('Random Seleted Digits')

