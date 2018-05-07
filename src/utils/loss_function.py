import numpy as np

class LossFunction(object):
	def __call__(self, y_actual, y_preficted):
		raise NotImplementedError()

	def gradient(self, y_actual, y_preficted):s
		raise NotImplementedError()

class SquaredLoss(LossFunction):
	def __call__(self, y_actual, y_preficted):
		return 0.5 * np.power((y_actual - y_preficted), 2)

	def gradient(self, y_actual, y_preficted):
		return -(y_actual - y_preficted)