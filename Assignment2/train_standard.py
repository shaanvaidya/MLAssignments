import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

data = pd.read_csv('train.csv')
data = clean_data(data, True)
y = data['salary']
x = data
del x['salary']

test = pd.read_csv('kaggle_test_data.csv')
ide = test['id']
test = clean_data(test, False)

gnb = GaussianNB()
gnb.fit(x,y)
out1 = gnb.predict(test)

lr = LogisticRegression()
lr.fit(x, y)
out2 = lr.predict(test)

rf = RandomForestClassifier(n_estimators=200)
rf.fit(x, y)
out3 = rf.predict(test)

# neigh = KNeighborsClassifier(n_neighbors=2)
# neigh.fit(x, y) 
# out3 = neigh.predict(test)

file1 = open('predictions_1.csv', 'w')
file2 = open('predictions_2.csv', 'w')
file3 = open('predictions_3.csv', 'w')

file1.write("id,salary\n")
file2.write("id,salary\n")
file3.write("id,salary\n")

for i in range(len(out1)):
	a1 = str(ide.iloc[i]) + ',' + str(out1[i]) + '\n'
	a2 = str(ide.iloc[i]) + ',' + str(out2[i]) + '\n'
	a3 = str(ide.iloc[i]) + ',' + str(out3[i]) + '\n'
	
	file1.write(a1)
	file2.write(a2)
	file3.write(a3)

file1.close()
file2.close()
file3.close()