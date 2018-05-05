from unsupervised import KMeans
import pandas as pd
import numpy as np

def testKMeans():
	dataset = pd.read_csv('data/iris.csv')
	dataset = dataset.loc[:, dataset.columns != 'Species']
	X = dataset.as_matrix()

	model = KMeans(n_clusters=3, iterations=20, live_plot=True)
	predicted_label = model.predict(X)

if __name__ == '__main__':
	testKMeans()

