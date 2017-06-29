import csv
import numpy as np 
import os

Lambda = 1
alpha = 0.0001
NoOfIterations = 50000

def loadData(filename):
	relativePath = os.path.dirname(os.path.abspath(__file__))
	Data = np.genfromtxt(os.path.join(relativePath,filename), delimiter = ',')
	return Data[1:]		#to remove the headings

#feature normalisation
def scale(x):
	mu = np.mean(x, axis = 0)
	std = np.std(x, axis = 0)
	x = (x - mu)/std
	return x, mu, std 		#mu, std passed for further use while prediction

#feature matrix
def createFeatureMatrix(Data):
	Data1 = np.square(Data)		#the square term
	Phi = np.c_[Data, Data1]
	Phi, mu, std = scale(Phi) 
	return np.c_[np.ones(np.shape(Data)[0]), Phi], mu, std 	#padded with ones; mu, std sent along

#the main algorithm
def GradientDescent(Phi, Y, alpha, NoOfIterations, Lambda):
	Y = np.matrix(Y).transpose()
	Phi = np.matrix(Phi)
	PhiT = Phi.transpose()
	m, n = np.shape(Phi)
	W = np.matrix(np.ones(n)).transpose()	#parameter initialisation
	i = 0
	while(i < NoOfIterations):
		PhiW = np.dot(Phi,W)
		W[0] = W[0] - alpha*np.dot((PhiW - Y).transpose(),Phi[:,0])		#not penalising the constant term
		W[1:] = W[1:] - alpha*(np.dot(Phi[:,1:].transpose(), (PhiW - Y)) + Lambda*W[1:])
		i = i + 1
	return W

#GD algorithm for p-norm
def GradientDescentP(Phi, Y, alpha, NoOfIterations, Lambda, p):
	Y = np.matrix(Y).transpose()
	Phi = np.matrix(Phi)
	PhiT = Phi.transpose()
	m, n = np.shape(Phi)
	W = np.matrix(np.ones(n)).transpose()
	i = 0
	while(i < NoOfIterations):
		PhiW = np.dot(Phi,W)
		W[0] = W[0] - alpha*np.dot((PhiW - Y).transpose(),Phi[:,0])
		W[1:] = W[1:] - alpha*(2*np.dot(Phi[:,1:].transpose(), (PhiW - Y)) + p*Lambda*abs(np.array(W[1:]))**(p-1))
		i = i + 1
	return W

#closed form solution
def NormalEquation(Phi, Y, Lambda):
	Y = np.matrix(Y).transpose()
	Phi = np.matrix(Phi)
	m, n = np.shape(Phi)
	PhiT = Phi.transpose()
	L = np.matrix(np.identity(n), copy = False)
	L[0,0] = 0
	A = np.dot(np.linalg.inv(np.dot(PhiT,Phi)+Lambda*L), PhiT)
	return np.dot(A,Y)

#predict the MEDV value using the W provided
def predict(TestData, Data, W, mu, std):
	TestData1 = np.square(TestData)
	PhiTest = np.c_[TestData, TestData1]
	PhiTest = (PhiTest - mu)/std 		#using the same mu and std
	PhiTest = np.c_[np.ones(np.shape(TestData)[0]), PhiTest]
	prediction = np.dot(PhiTest,W)
	return prediction

Data = loadData('data/train.csv')
Y = Data[:,-1]
Data = Data[:,1:]
Phi, mu, std = createFeatureMatrix(Data[:,:-1])

W = GradientDescent(Phi, Y, alpha, NoOfIterations, Lambda)
W1 = GradientDescentP(Phi, Y, alpha, NoOfIterations, Lambda, 1.2)
W2 = GradientDescentP(Phi, Y, alpha, NoOfIterations, Lambda, 1.5)
W3 = GradientDescentP(Phi, Y, alpha, NoOfIterations, Lambda, 1.7)
# Wformula = NormalEquation(Phi, Y, Lambda)

# print(np.linalg.norm(W - Wformula)) #for comparing gradient descent solution and the closed form solution

TestData = loadData('data/test.csv')
ID = TestData[:,0]

prediction = predict(TestData[:,1:], Data, W, mu, std)
prediction = np.c_[ID, prediction]
# predictionFormula = predict(TestData[:,1:], Data, W, mu, std)
# predictionFormula = np.c_[ID, predictionFormula]
prediction1 = predict(TestData[:,1:], Data, W1, mu, std)
prediction1 = np.c_[ID, prediction1]
prediction2 = predict(TestData[:,1:], Data, W2, mu, std)
prediction2 = np.c_[ID, prediction2]
prediction3 = predict(TestData[:,1:], Data, W3, mu, std)
prediction3 = np.c_[ID, prediction3]

np.savetxt("output.csv", prediction, delimiter=",", fmt=['%.0f', '%.5f'], header="ID,MEDV", comments='')	
# np.savetxt("output_Formula.csv", predictionFormula, delimiter=",", fmt=['%.0f', '%.5f'], header="ID,MEDV", comments='')
np.savetxt("output_p1.csv", prediction1, delimiter=",", fmt=['%.0f', '%.5f'], header="ID,MEDV", comments='')
np.savetxt("output_p2.csv", prediction2, delimiter=",", fmt=['%.0f', '%.5f'], header="ID,MEDV", comments='')
np.savetxt("output_p3.csv", prediction3, delimiter=",", fmt=['%.0f', '%.5f'], header="ID,MEDV", comments='')
#fmt for data types, comments to remove that extra '#' for comments