import numpy as np
import math, random

MAX_DISTANCE = np.finfo(np.float64).max

class KMeans(object):
	def __init__(self, n_clusters=3, iterations=30, live_plot=False):
		self.n_clusters = n_clusters
		self.iterations = iterations
		self.live_plot = live_plot

		if self.live_plot:
			global plt
			from matplotlib import pyplot as plt
			plt.style.use('ggplot')

	def predict(self, X):
		centroids = self._setRandomCentroids(X)
		for iter_n in range(self.iterations):
			# Cluster is an array of arrays in which each array contains indices of the points belonging to that cluster
			clusters = self._createClusters(X, centroids)

			centroids = self._adjustCentroids(X, clusters)
			if self.live_plot:
				self._prepareDataNShow(X, centroids, clusters, iter_n == (self.iterations-1))

		return self._getPredictedLabels(X, clusters)

	def _getPredictedLabels(self, X, clusters):
		predicted_labels = np.zeros(X.shape[0], dtype=np.int)
		for cluster_index, cluster in enumerate(clusters):
			predicted_labels[cluster] = cluster_index
		return predicted_labels


	def _setRandomCentroids(self, X):
		n_samples, n_features = np.shape(X)
		centroids = np.zeros((self.n_clusters, n_features))
		for i in range(self.n_clusters):
			centroid = X[np.random.random_integers(0, n_samples-1)]
			centroids[i] = centroid
		return centroids

	def _createClusters(self, X, centroids):
		clusters = [[] for _ in range(self.n_clusters)]
		for sample_index, sample in enumerate(X):
			closest_centroid_index = self._getClosestCentroid(sample, centroids)
			clusters[closest_centroid_index].append(sample_index)
		return clusters


	def _getClosestCentroid(self, data_point, centroids):
		closest_distance = MAX_DISTANCE
		closest_centroid_index = 0
		for index, centroid in enumerate(centroids):
			distance = self._getEuclideanDistance(data_point, centroid)
			if distance < closest_distance:
				closest_distance = distance
				closest_centroid_index = index
		return closest_centroid_index

	def _getEuclideanDistance(self, point_1, point_2):
		return np.sqrt(np.sum((point_1 - point_2)**2))

	def _adjustCentroids(self, X, clusters):
		n_samples, n_features = np.shape(X)
		new_centroids = np.zeros((self.n_clusters, n_features))
		for cluster_index, cluster in enumerate(clusters):
			new_centroid = np.mean(X[cluster], axis=0)
			new_centroids[cluster_index] = new_centroid
		return new_centroids

	def _prepareDataNShow(self, X, centroids, clusters, continue_showing):
		rows, columns = X.shape
		if columns < 2:
			print "There should be atleast two features in the data for scatter plot"
			return False
		feature_one = X[:,0]
		feature_two = X[:,1]

		point_colors = np.zeros(rows)
		point_sizes = np.ones(rows) * (2**4)

		for cluster_index, cluster in enumerate(clusters):
			point_colors[cluster] = (cluster_index + 3) * 1.0

		centroid_feature_one = centroids[:, 0]
		centroid_feature_two = centroids[:, 1]

		feature_one = np.concatenate([feature_one, centroid_feature_one])
		feature_two = np.concatenate([feature_two, centroid_feature_two])

		point_colors = np.concatenate([point_colors, [ i * 1.0 for i in range(3, self.n_clusters + 3)]])
		point_sizes = np.concatenate([point_sizes, [2**6] * self.n_clusters])

		self._showLivePlot(feature_one, feature_two, point_colors, point_sizes, continue_showing)


	def _showLivePlot(self, feature_one, feature_two, point_colors, point_sizes, continue_showing=False):
		if not self.live_plot:
			return False

		fig = plt.figure()
		axis = fig.add_subplot(111)
		axis.scatter(feature_one, feature_two, c=point_colors, s=point_sizes, alpha = 1.0)
		plt.tight_layout()
		plt.draw()
		plt.pause(0.1)
		if continue_showing:
			plt.show()
		else:
			plt.close(fig)
