import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 

fields_num = ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",)

fields = (
	("workclass", (" Private", " Self-emp-not-inc", " Self-emp-inc", " Federal-gov", " Local-gov", " State-gov", " Without-pay", " Never-worked")), 
	("marital-status", (" Married-civ-spouse", " Divorced", " Never-married", " Separated", " Widowed", " Married-spouse-absent", " Married-AF-spouse")), 
	("occupation", (" Tech-support", " Craft-repair", " Other-service", " Sales", " Exec-managerial", " Prof-specialty", " Handlers-cleaners", " Machine-op-inspct", " Adm-clerical", " Farming-fishing", " Transport-moving", " Priv-house-serv", " Protective-serv", " Armed-Forces")), 
	("relationship", (" Wife", " Own-child", " Husband", " Not-in-family", " Other-relative", " Unmarried")), 
	("race", (" White", " Asian-Pac-Islander", " Amer-Indian-Eskimo", " Other", " Black")), 
	("sex", (" Female", " Male")),
)

means = []
stds = []

def clean_data(data, isTrain):
	data_new = data
	if isTrain:
		for i in range(len(fields_num)):
			temp = data_new[fields_num[i]]
			data_new[fields_num[i]] = (temp - temp.mean())/temp.std()
			means.append(temp.mean())
			stds.append(temp.std())
	else:
		for i in range(len(fields_num)):
			temp = data_new[fields_num[i]]
			data_new[fields_num[i]] = (temp - means[i])/stds[i]

	for i in range(len(fields)):
		for j in range(len(fields[i][1])):
			label = fields[i][0] + '_' + fields[i][1][j]
			data_new[label] = data_new[fields[i][0]] == fields[i][1][j]
			data_new[label] = data_new[label].astype(int)
		del data_new[fields[i][0]]
	del data_new['native-country']
	del data_new['education']
	del data_new['id']
	return data_new

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_dash(z):
	return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def tanh(z):
	return np.tanh(z)

def tanh_dash(z):
	return 1 - np.multiply(tanh(z), tanh(z))

def relu(z):
	return np.maximum(z, 0)

def relu_dash(z):
	return float(z > 0)

class MyNeuralNetwork(object):

	def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, epochs=10):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.zs = [np.zeros((y,1)) for y in sizes[1:]]
		self.activations = [np.zeros((y,1)) for y in sizes]
		self.mini_batch_size = mini_batch_size
		self.epochs = epochs
		self.eta = learning_rate

	def fit(self, training_data, validation_data=None):
		for epoch in range(self.epochs):
			training_data = training_data.sample(frac=1).reset_index(drop=True)
			mini_batches = [training_data[k:k + self.mini_batch_size] for k in range(0, len(training_data), self.mini_batch_size)]

			for mini_batch in mini_batches:
				nabla_b = [np.zeros(bias.shape) for bias in self.biases]
				nabla_w = [np.zeros(weight.shape) for weight in self.weights]
				for i in range(len(mini_batch)):
					x = mini_batch.iloc[i]
					y = x['salary']
					del x['salary']
					self.forward_propagate(x)
					delta_nabla_b, delta_nabla_w = self.back_propagate(y)
					nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
					nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

				self.weights = [w - (self.eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
				self.biases = [b - (self.eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
		
		return self.weights, self.biases

			# if validation_data!=None:
				# accuracy = self.validate(validation_data) / 100.0
			# else:
				# print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))
	def validate(self, validation_data):
		validation_results = []
		for i in range(len(validation_data)):
			x = validation_data.iloc[i]
			y = x['salary']
			del x['salary']
			validation_results.append(self.predict(x) == y)
		return sum(result for result in validation_results)

	def predict(self, x):
		self.forward_propagate(x)
		return (self.activations[-1] > 0.5)

	def predictOnTest(self,test):
		y = []
		for i in range(len(test)):
			x = test.iloc[i]
			self.forward_propagate(x)
			y.append(int(self.activations[-1] > 0.5))	
		return y

	def forward_propagate(self, x):
		self.activations[0] = np.matrix(x).transpose()
		for i in range(self.num_layers-1):
			self.zs[i] = (self.weights[i].dot(self.activations[i]) + self.biases[i])
			self.activations[i+1] = tanh(self.zs[i])

	def back_propagate(self, y):
		nabla_b = [np.zeros(bias.shape) for bias in self.biases]
		nabla_w = [np.zeros(weight.shape) for weight in self.weights]

		error = (self.activations[-1] - y) * tanh_dash(self.zs[-1])
		nabla_b[-1] = error
		nabla_w[-1] = error.dot(self.activations[-2].transpose())

		for l in range(2,self.num_layers):
			error = np.multiply(self.weights[-l + 1].transpose().dot(error), tanh_dash(self.zs[-l]))
			nabla_b[-l] = error
			nabla_w[-l] = error.dot(self.activations[-l - 1].transpose())

		return nabla_b, nabla_w

data = pd.read_csv('train.csv')
# data = data.sample(frac=0.01).reset_index(drop=True)
# test = pd.read_csv('test.csv')
# ide = test['id']
data = clean_data(data, True)
# test = clean_data(test, False)
mini_batch_size = 16
net = MyNeuralNetwork(sizes=[48,30,20,1])
weights, biases = net.fit(data)

meansandstds = np.c_[means,stds]
np.savetxt("meansandstds.txt", meansandstds, delimiter=",", fmt=['%.5f', '%.5f'], header="means,stds", comments='')	
sizes = [48,30,20,1]
np.savetxt("sizes.txt", sizes, delimiter=",", fmt=['%.0f'], header="sizes", comments='')	

out_weights = []
for weight in weights:
	out_weights.append(weight.reshape(1,weight.shape[0]*weight.shape[1]))
out_biases = biases

file = open('weights.txt', 'w')
for weight in out_weights:
	for i in range(weight.shape[1]):
		a = str(weight[0,i]) + '\n'
		file.write(a)

file.close()

file = open('biases.txt', 'w')
for bias in out_biases:
	for i in range(len(bias)):
		a = str(bias[i,0]) + '\n'
		file.write(a)		

file.close()

# y = net.predictOnTest(test)
# for i in range(len(y)):
# 	a = str(ide.iloc[i]) + ',' + str(y[i])
# 	print(a)

# validation = clean_data(validation, True)
#######################################################
# fig = plt.figure(figsize=(20,15))
# cols = 5	
# rows = int(float(data.shape[1]) / cols)
# for i, column in enumerate(data.columns):
#     ax = fig.add_subplot(rows, cols, i + 1)
#     ax.set_title(column)
#     if data.dtypes[column] == np.object:
#         data[column].value_counts().plot(kind="bar", axes=ax)
#     else:
#         data[column].hist(axes=ax)
#         plt.xticks(rotation="vertical")
# plt.subplots_adjust(hspace=0.7, wspace=0.2)
# plt.show()

########################################################
