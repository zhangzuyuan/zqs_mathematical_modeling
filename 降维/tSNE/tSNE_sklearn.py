#手写数字数据集的可视化：
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

#https://blog.csdn.net/tszupup/article/details/84997804

# 加载数据
def get_data():
	"""
	:return: 数据集、标签、样本数量、特征数量
	"""
	digits = datasets.load_digits(n_class=10)
	data = digits.data		# 图片特征
	label = digits.target		# 图片标签
	n_samples, n_features = data.shape		# 数据集的形状
	return data, label, n_samples, n_features

# 对样本进行预处理并画图
def plot_embedding(data, label, title):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	fig = plt.figure()		# 创建图形实例
	ax = plt.subplot(111)		# 创建子图
	# 遍历所有样本
	for i in range(data.shape[0]):
		# 在图中为每个数据点画出标签
		plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
				 fontdict={'weight': 'bold', 'size': 7})
	plt.xticks()		# 指定坐标的刻度
	plt.yticks()
	plt.title(title, fontsize=14)
	# 返回值
	return fig


# 主函数，执行t-SNE降维
def main():
	data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
	print('Starting compute t-SNE Embedding...')
	ts = TSNE(n_components=2, init='pca', random_state=0)
	# t-SNE降维
	reslut = ts.fit_transform(data)
	# 调用函数，绘制图像
	fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
	# 显示图像
	plt.show()


# 主函数
if __name__ == '__main__':
	main()