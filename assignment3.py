import numpy as np
import math


### Assignment 3 ###

class KNN:
	def __init__(self, k):
		# KNN state here
		# Feel free to add methods
		self.k = k
		self.instances = None
		self.labels = None

	def distance(self, feature_a, feature_b):
		diffs = (feature_a - feature_b) ** 2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		# training logic here
		# input is an array of features and labels
		self.instances = X
		self.labels = y

	def predict(self, X):
		# Run model here
		predictions = []

		for instance in X:
			distance_to_neighbours = [self.distance(instance, row) for row in self.instances]
			k_closest_neighbours = np.argsort(distance_to_neighbours)[:self.k]
			closest_neighbours_labels = [self.labels[i] for i in k_closest_neighbours]
			neighbour_label, count = np.unique(closest_neighbours_labels, return_counts=True)
			index_of_most_common_neighbour = np.argmax(count)
			predictions.append(neighbour_label[index_of_most_common_neighbour])

		return np.array(predictions)


class Perceptron:
	def __init__(self, w, b, lr):
		# Perceptron state here, input initial weight matrix
		# Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		# training logic here
		# input is array of features and labels

		training_size = len(X)

		for step in range(steps):
			# Get the index of the current data point we are processing
			index = step % training_size
			instance = X[index]

			activation_output = np.dot(self.w, instance) + self.b
			step_function_output = 1 if activation_output > 0 else 0

			# If the prediction is mis-labeled, run the perceptron
			if step_function_output != y[index]:
				# w ← w + (η di xi)
				self.w = self.w + (self.lr * (y[index] - step_function_output) * instance)

	def predict(self, X):
		# Run model here
		# Return array of predictions where there is one prediction for each set of features

		result = []

		for x in X:
			activation_function_output = np.dot(self.w, x) + self.b
			step_function_output = 1 if activation_function_output > 0 else 0
			result.append(step_function_output)

		return np.array(result)


class ID3:
	def __init__(self, nbins, data_range):
		# Decision tree state here
		# Feel free to add methods
		self.bin_size = nbins
		self.range = data_range
		self.tree = None

	def preprocess(self, data):
		# Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size * norm_data).astype(int)
		return categorical_data

	# This method calculates the Shannon Entropy for a given set of labels.
	# For Shannon Entropy, we consider only the bins of the labels, passing the label list here suffices.
	def calculate_entropy(self, labels):
		total_count = len(labels)
		unique_labels, counts = np.unique(labels, return_counts=True)
		entropy = sum((-(c / total_count) * math.log2(c / total_count)) if c > 0 else 0 for c in counts)
		return entropy

	# This method returns the info_gain of a column for a given feature list.
	# Bin the labels corresponding to unique feature bins for a given column, calculate the entropy and info gain.
	def calculate_information_gain(self, instances, column, labels):
		feature = [instance[column] for instance in instances]

		unique_bins, counts = np.unique(feature, return_counts=True)
		total_count = sum(counts)

		bin_entropies = []
		for x in unique_bins:
			indexes = np.where(feature == x)[0]
			corresponding_labels = [labels[i] for i in indexes]
			bin_entropy = self.calculate_entropy(corresponding_labels)
			bin_entropies.append(bin_entropy)

		info_gain = sum(((counts[i] / total_count) * bin_entropies[i]) for i in range(len(bin_entropies)))
		return info_gain

	# This method returns the column index of the feature with best gain.
	# Calculate the gain for every column that has not already been considered for tree split, sort them
	# Return the column index of the highest gain
	def get_best_gain_column(self, instances, labels, columns_to_process):
		total_entropy = self.calculate_entropy(labels)
		gain = {}
		for column in columns_to_process:
			gain[column] = total_entropy - self.calculate_information_gain(instances, column, labels)
		return max(gain, key=gain.get)

	# This method returns the most common value for a given array
	def get_mode(self, values):
		bins, counts = np.unique(values, return_counts=True)
		return bins[np.argmax(counts)]

	# This is the main function of the algorithm training.
	# We recursively calculate the nodes and append to tree
	# Features is the set of datapoints that we are processing, labels are their corresponding subset from the y
	# columns_to_process maintains the columns that have not yet been visited
	# parent_examples holds the values of the previously selected best column from the parent subset
	def decision_tree_learning(self, instances, labels, columns_to_process, parent_examples=None):
		# If the examples is empty, return the PLURALITY_VALUES(parent_examples)
		# Parent_Examples are values of the best gain column on which we have split
		if len(instances) == 0:
			return self.get_mode(parent_examples)
		# Else If All examples have the same classification, return the classification
		elif len(np.unique(labels)) == 1:
			return labels[0]
		# Else If there are no more columns to process, then return the most common label for the data set
		elif len(columns_to_process) == 0:
			return self.get_mode(labels)

		# Else
		# A ← argmax(a ∈ attributes) IMPORTANCE(a, examples)
		best_column = self.get_best_gain_column(instances, labels, columns_to_process)
		# Remove the best column from columns to process, since it is already selected
		columns_to_process.remove(best_column)

		# Tree will be constructed using a dictionary
		# If the value of a key is a dictionary, then the key is a column number
		# If the value of a key is an integer/string, then the key is the value on which we need to split
		tree = {best_column: {}}

		best_column_values = [instance[best_column] for instance in instances]

		# Calculate the exs ← {e : e ∈ examples and e.A = vk} - Bins of unique values of the selected column
		bins = np.unique(best_column_values)

		# If all the examples are same - return the PLURALITY_VALUE(classification) - most common label for the features
		if len(bins) == 1:
			return self.get_mode(labels)

		for bin_value in bins:
			# Create the subset of features for each bin. Obtain the corresponding labels.
			indexes = [i for i in range(len(best_column_values)) if best_column_values[i] == bin_value]
			instance_subset = [instances[i] for i in indexes]
			label_subset = [labels[i] for i in indexes]

			# subtree ← DECISION-TREE-LEARNING(exs, attributes − A, examples)
			subtree = self.decision_tree_learning(instance_subset, label_subset, columns_to_process, labels)
			# Add a branch to tree with label (A = vk) and subtree subtree
			tree[best_column][bin_value] = subtree

		return tree

	def train(self, X, y):
		# training logic here
		# input is array of features and labels
		categorical_data = self.preprocess(X)

		self.tree = self.decision_tree_learning(categorical_data, y, list(range(categorical_data.shape[1])))

	def predict(self, X):
		# Run model here
		# Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)

		predictions = []

		for data in categorical_data:
			# Get the root node of the built decision tree
			# Since I have used a dictionary, feeding in the entire dictionary as root node
			root_node = self.tree

			# Iteratively traverse through the nested dictionary to obtain the prediction
			# While a leaf has not been reached, do
			while 1:
				# If the object is not a dictionary, return it as is, since we have reached a value
				if not isinstance(root_node, dict):
					predictions.append(root_node)
					break
				else:
					# Get the key for the dictionary of the node that we have reached
					# This is the column index whose value we need to test
					column_index = next(iter(root_node))

					# Get the value from the column in the testing instance
					feature_value = data[column_index]

					# If the value is present as a key in the dictionary of the node, then move to that node
					if feature_value in root_node[column_index]:
						root_node = root_node[column_index][feature_value]
					# Else, we have reached a value that we have not come across during the model building
					# Abort the execution for the current data
					else:
						predictions.append(None)
						break

		return np.array(predictions)

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

