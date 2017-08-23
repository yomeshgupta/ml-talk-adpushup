import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pylab import plot, show, xlabel, ylabel

def get_data():
	# load data
	df = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
	data = df.as_matrix()

	# all data containing only features
	X = data[:,:-1]
	# all outputs
	y = data[:,-1:]
	
	noOfFeatures = X.shape[1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=0)

	return X_train, y_train, X_test, y_test, noOfFeatures

def computeCost(X, y, theta):
	pass

def gradientDescent(X, y, theta, alpha, num_iters):
	pass

def featureNormalize(X):
	pass

def predict(inputVector, theta):
	return np.dot(inputVector, theta.T)

def plot2DGraph(x, y, xLabel, yLabel):
	plot(np.arange(x), y)
	xlabel(xLabel)
	ylabel(yLabel)
	return show()

def run():
	print("Hello world")

if __name__ == '__main__':
	run()