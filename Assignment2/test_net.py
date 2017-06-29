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

meansandstds = pd.read_csv('meansandstds.txt')
means = np.array(meansandstds['means'])
stds = np.array(meansandstds['means'])

def clean_data(data):
	data_new = data
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

def tanh(z):
	return np.tanh(z)

def tanh_dash(z):
	return 1 - np.multiply(tanh(z), tanh(z))

class MyNeuralNetwork(object):

	def __init__(self,weights,biases,sizes=list()):
		self.weights = weights
		self.biases = biases
		self.zs = [np.zeros((y,1)) for y in sizes[1:]]
		self.activations = [np.zeros((y,1)) for y in sizes]
		self.num_layers = len(sizes)

	def predict(self,test):
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

lines = []
sizes = [48,30,20,1]

with open('weights.txt', 'r') as file:
	for line in file:
		line = line.strip()
		lines.append(line)
lines = list(map(float,lines))

# for line in lines:
# 	line = float(line)

lengths = list(zip(sizes[1:], sizes[:-1]))
otherlengths = [x*y for x,y in lengths]
processedweights = []

i = 0
otherweights = []
for l in otherlengths:
	otherweights.append(lines[i:i+l])
	i = i + l

weights = []
for w in range(len(otherweights)):
	weights.append(np.array(otherweights[w]).reshape(sizes[w+1], sizes[w]))

lines = []
with open('biases.txt', 'r') as file:
	for line in file:
		line = line.strip()
		lines.append(line)


# for line in lines:
	# line = float(line)
lines = list(map(float,lines))

biases = []
otherlengths = sizes[1:]
otherbiases = []

i = 0
for l in otherlengths:
	otherbiases.append(lines[i:i+l]) 
	i = i + l

for l in range(len(otherlengths)):
	biases.append(np.array(otherbiases[l]).reshape(sizes[l+1], 1))

# print(biases)
# print(weights)
# a = [1,2,3]
# print(a)
test = pd.read_csv('kaggle_test_data.csv')
ide = test['id']
test = clean_data(test)
mini_batch_size = 16
net = MyNeuralNetwork(weights = weights, biases = biases, sizes=[48,30,20,1])
y = net.predict(test)
file = open('predictions.csv', 'w')
file.write("id,salary\n")

for i in range(len(y)):
	a = str(ide.iloc[i]) + ',' + str(y[i]) + '\n'
	file.write(a)

file.close()
