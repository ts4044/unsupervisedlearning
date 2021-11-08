import numpy as np


### Assignment 3 ###

class KNN:
	def __init__(self, k):
		# KNN state here
		# Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB) ** 2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		# training logic here
		# input is an array of features and labels
		self.features = X
		self.labels = y

	def predict(self, X):
		# Run model here
		predictions = []

		for predict in X:
			# Find the Euclidian distance with each of the rows in features and form an array
			neighbours = [self.distance(predict, row) for row in self.features]

			# Sort the array and obtain the index of the top 'k' neighbours
			k_sorted_neighbours = np.argsort(neighbours)[:self.k]

			# Based on the indexes of the top k neighbours, get their labels
			k_neighbours_labels = [self.labels[i] for i in k_sorted_neighbours]

			# Count the labels
			k_nearest_neighbours, count = np.unique(k_neighbours_labels, return_counts=True)

			# Get the index of the label that has the maximum count
			index_of_mode = np.argmax(count)

			# Add the predicted label to the result list
			predictions.append(k_nearest_neighbours[index_of_mode])

		# Return array of predictions where there is one prediction for each set of features
		return np.array(predictions)


class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		return None


class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		return None


### Assignment 4 ###

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None


class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum(axis=1))

	def train(self, X):
		#training logic here
		#input is array of features (no labels)


		return self.cluster
		#return array with cluster id corresponding to each item in dataset


class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		#training logic here
		#input is array of features (no labels)


		return self.cluster
		#return array with cluster id corresponding to each item in dataset

