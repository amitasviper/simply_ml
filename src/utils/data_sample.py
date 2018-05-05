import numpy as np

class DataSample(object):
	def __init__(self, x_data, y_data=None):
		self.data_points = None

		if y_data == None or len(y_data) == 0:
			print "y_data is empty. Assigning dummy target value to all x_data points"
			y_data = [0] * len(x_data)

		if len(x_data) != len(y_data):
			print 'The size of x_data and y_data does not match. Stripping the extra part. This may result in unexpected results.'
			min_len = min(len(x_data), len(y_data))
			x_data = x_data[:min_len]
			y_data = y_data[:min_len]

		self.coupleDataPoints(x_data, y_data)


	def coupleDataPoints(self, x_data, y_data):
		self.data_points = []
		for x_data_point, y_data_point in zip(x_data, y_data):
			data_point = {'x': np.array(x_data_point), 'target': y_data_point}
			self.data_points.append(data_point)