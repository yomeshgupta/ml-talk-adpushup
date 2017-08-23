import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pylab import plot, show, xlabel, ylabel

def get_data():
	# load data
	df = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
	data = df.as_matrix()

	break_point = 170 # data break_point
	
	# all data containing only features
	X = data[:,:-1]
	# all outputs
	y = data[:,-1:]
	
	noOfFeatures = X.shape[1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=0)

	return X_train, y_train, X_test, y_test, noOfFeatures

def computeCost(X, y, theta):
	m = y.shape[0]
	predictions = X.dot(theta)
	sqErrors = (predictions - y)
	J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
	return J

def gradientDescent(X, y, theta, alpha, num_iters):
	cost = np.zeros(num_iters, dtype=float)
	m = len(X)

	for i in range(num_iters):
		error = (X * theta) - y
		newDecrement = (alpha * (1/m) * np.dot(error.T, X))
		theta = theta - newDecrement.T

		cost[i] = computeCost(X, y, theta)

	return theta, cost

def featureNormalize(X):
	mean_r = []
	std_r = []

	X_norm = X
	n_c = X.shape[1]
	for i in range(n_c):
		meanOfCurrentFeatureInX = np.mean(X[:, i])
		stdOfCurrentFeatureInX = np.std(X[:, i])
		mean_r.append(meanOfCurrentFeatureInX)
		std_r.append(stdOfCurrentFeatureInX)
		if (stdOfCurrentFeatureInX == 0.0 or stdOfCurrentFeatureInX == 0):
			X_norm[:, i] = X_norm[:, i]
		else:
			X_norm[:, i] = (X_norm[:, i] - meanOfCurrentFeatureInX) / stdOfCurrentFeatureInX

	return X_norm, mean_r, std_r

def predict(inputVector, theta):
	return np.dot(inputVector, theta.T)

def plot2DGraph(x, y, xLabel, yLabel):
	plot(np.arange(x), y)
	xlabel(xLabel)
	ylabel(yLabel)
	return show()

def run():
	alpha = 0.01
	num_iters = 400
	X_train, y_train, X_test, y_test, noOfFeatures = get_data()

	X_train_norm, mu, sigma = featureNormalize(X_train)
	X_test_norm, test_mu, test_sigma = featureNormalize(X_test)
	theta = np.matrix(np.zeros((noOfFeatures + 1, 1), dtype=np.float))

	# Adding intercept term to X
	X_train_norm = np.column_stack((np.ones(len(X_train_norm), dtype=np.float), X_train_norm))
	X_test_norm = np.column_stack((np.ones(len(X_test_norm), dtype=np.float), X_test_norm))

	# perform linear regression on the data set
	computed_theta, J_history = gradientDescent(X_train_norm, y_train, theta, alpha, num_iters)

	# get the cost (error) of the model
	cost = computeCost(X_train_norm, y_train, computed_theta)
	predict = np.dot(X_test_norm, computed_theta)

	plot2DGraph(num_iters, J_history, 'Iterations', 'Cost Function')

	print(cost)
	print(y_test[0:5])
	print(predict[0:5])

if __name__ == '__main__':
	run()