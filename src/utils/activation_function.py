import numpy as np

class ActivationFunction(object):
	def __call__(self, x):
		raise NotImplementedError()

	def gradient(self, x):
		raise NotImplementedError()

class Sigmoid(ActivationFunction):
	def __call__(self, x):
		return (1 / (1 + np.exp(-x)))

	def gradient(self, x):
		value = self.__call__(x)
		return value * (1 - value)

class ReLU(ActivationFunction):
	def __call__(self, x):
		return np.maximum(x, 0)

	def gradient(self, x):
		return np.minimum(1, self.__call__(x))

class Tanh(ActivationFunction):
	def __call__(self, x):
		return np.tanh(x)

	def gradient(self, x):
		return (1 - (x ** 2))

class SoftMax(ActivationFunction):
	def __call__(self, x):
		exp_raised = np.exp(x)
		return (exp_raised / np.sum(exp_raised))