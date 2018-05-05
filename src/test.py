from unsupervised import KMeans
from utils import DataSample
import math, random

if __name__ == '__main__':
	x = []
	for i in range(500):
		if i < 250:
			a = 10 + 19 * math.cos(2*math.pi*random.random())
			b = 30 + 19 * math.sin(2*math.pi*random.random())
			x.append([a, b])
		else:
			a = 66 + 32 * math.cos(2*math.pi*random.random())
			b = 58 + 32 * math.sin(2*math.pi*random.random())
			x.append([a, b])

	data = DataSample(x).data_points
	model = KMeans(n_clusters=2, iterations=40)
	model.fit(data)

