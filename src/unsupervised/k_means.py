import numpy as np
import math, random
from matplotlib import pyplot as plt
plt.style.use('ggplot')

MAX_DISTANCE = np.finfo(np.float64).max

from utils import DataSample 

class KMeans(object):
	def __init__(self, n_clusters=3, iterations=30):
		self.n_clusters = n_clusters
		self.iterations = iterations
		self.data_dimension = 0

	def fit(self, data_sample):
		self.data_sample = data_sample
		self.data_dimension = len(self.data_sample[0]['x'])
		self._setDefaultMeans()
		for iter_n in range(self.iterations):
			self._train(iter_n)

	def _setDefaultMeans(self):
		self.means = []
		for i in range(self.n_clusters):
			mean_info = {'label': i, 'mean': np.random.rand(self.data_dimension)}
			self.means.append(mean_info)
		plt.show()

	def _getDataDimensionsSumStruct(self):
		data_dim_sum = {}
		for i in range(self.n_clusters):
			data_dim_sum[self.means[i]['label']] = [np.zeros(self.data_dimension), 0]
		return data_dim_sum


	def _train(self, iteration):
		data_points_sum = self._getDataDimensionsSumStruct()
		colors = ['red', 'green', 'blue', 'purple', 'yellow']
		x_points = []
		y_points = []
		size = []
		color_points = []
		for data_point in self.data_sample:
			x = data_point['x']
			mean_label = self._getClosestMean(x)
			data_point['target'] = mean_label
			# print "Previous value : ", data_points_sum[mean_label][0]
			# print "Adding x : ", x
			data_points_sum[mean_label][0] += x
			data_points_sum[mean_label][1] += 1
			# print data_points_sum
			x_points.append(x[0])
			y_points.append(x[1])
			color_points.append(colors[mean_label])
			size.append(2**2)

		adjusted_means = []
		clustered_data_points = []
		for i in range(self.n_clusters):
			# print "Hello : ", (data_points_sum[self.means[i]['label']][0] / (1.0 * data_points_sum[self.means[i]['label']][1]))
			if data_points_sum[self.means[i]['label']][1] != 0:
				self.means[i]['mean'] = (data_points_sum[self.means[i]['label']][0] / (1.0 * data_points_sum[self.means[i]['label']][1]))
			x_points.append(self.means[i]['mean'][0])
			y_points.append(self.means[i]['mean'][1])
			color_points.append(colors[self.means[i]['label']])
			size.append(2**6)

		print len(x_points)
		# print x_points
		# plt.scatter(x_points, y_points, c=color_points, s=size)
		# plt.show()
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.scatter(x_points, y_points, c=color_points, s=size)


		plt.tight_layout()

		plt.draw()
		plt.pause(0.4)
		plt.close(fig)




	def _getClosestMean(self, x):
		distance = MAX_DISTANCE
		closest_mean = 0
		for i in range(len(self.means)):
			dist = self._getDistance(x, self.means[i]['mean'])
			if dist < distance:
				distance = dist
				closest_mean = self.means[i]['label']
		return closest_mean


	def _getDistance(self, point_1, point_2):
		return np.linalg.norm(point_1 - point_2)

	def predict(self, data_sample):
		pass
