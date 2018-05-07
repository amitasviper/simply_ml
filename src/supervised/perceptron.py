import numpy as np
from utils import Sigmoid
from utils import SquaredLoss

class Perceptron(object):
	def __init__(self, max_iters=600, learning_rate=0.01, activation=Sigmoid, loss=SquaredLoss, live_plot=False):
		self.max_iters = max_iters
		self.learning_rate = learning_rate
		self.activation = activation()
		self.loss = loss()
		self.live_plot = live_plot
		if self.live_plot:
			global plt
			from matplotlib import pyplot as plt
			plt.style.use('ggplot')

	def _prepareLivePlot(self, iter_num, error_gradient):
		if not self.live_plot:
			return False

		if iter_num == 0:
			self.x_iter = []
			self.y_error = []
		self.x_iter.append(iter_num)
		self.y_error.append(error_gradient)

		self._showLivePlot(self.x_iter, self.y_error)

	def _showLivePlot(self, epoch, error_gradient, continue_showing=False):
		
		fig = plt.figure()
		axis = fig.add_subplot(111)
		axis.plot(epoch, error_gradient)
		plt.xlim(0, self.max_iters)
		# plt.ylim(ymin=-1, ymax=0.)

		plt.tight_layout()
		plt.draw()
		plt.pause(0.001)
		if continue_showing:
			plt.show()
		else:
			plt.close(fig)


	def fit(self, X, y):
		n_samples, n_features = np.shape(X)
		n_outputs = np.shape(y)[0]
		self.weights = np.random.uniform(-1, 1, (n_features, n_outputs))
		self.bias = np.zeros((1, n_outputs))
		for iteration in range(self.max_iters):
			linear_product = X.dot(self.weights) + self.bias
			y_predicted = self.activation(linear_product)

			error_gradient = self.activation.gradient(linear_product) * self.loss.gradient(y, y_predicted)

			gradient_per_weight = X.T.dot(error_gradient)
			gradient_per_bias = np.sum(error_gradient, axis=0, keepdims=True)

			self.weights  -= self.learning_rate * gradient_per_weight
			self.bias -= self.learning_rate  * gradient_per_bias
			self._prepareLivePlot(iteration, self.loss(y, y_predicted).mean())


	def predict(self, X):
		y_predicted = self.activation(X.dot(self.weights) + self.bias)
		return y_predicted