from supervised import Perceptron
from unsupervised import KMeans
import pandas as pd
import numpy as np

def testKMeans():
	dataset = pd.read_csv('data/iris.csv')
	dataset = dataset.loc[:, dataset.columns != 'Species']
	X = dataset.as_matrix()

	model = KMeans(n_clusters=3, iterations=20, live_plot=True)
	predicted_label = model.predict(X)

def testPerceptron():
	train_dataset = pd.read_csv('data/sonar.csv')
	train_X = train_dataset.iloc[:, :-1]
	train_y = train_dataset.iloc[:, -1]
	train_y = train_y.replace(['R'], 0)
	train_y = train_y.replace(['M'], 1)

	train_X = train_X.as_matrix()
	train_y = train_y.as_matrix()

	percep = Perceptron(max_iters=80, learning_rate=0.01,live_plot=True)
	percep.fit(train_X, train_y)



if __name__ == '__main__':
	# testKMeans()
	testPerceptron()

