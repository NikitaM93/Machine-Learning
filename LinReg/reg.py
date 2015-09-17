#Nikita Miroshnichenko
#CSC 246, Professor Gildea
#27554869

import numpy as np 
from numpy import linalg

class reg(object):
	def __init__(self, X, Y, numfeats, Lambda):
		#regression initialization.
		#weights are trained and saved in w.
		self.X=X
		self.Y=Y
		self.Lambda=Lambda
		self.w=np.array(numfeats+1)
		#self.weightMatrix=np.zeros([numfeats+1, 1]) 

	def train(self):
		#used for training weights.
		#formula: inverse(X*transpose(X)+lambda*I)*X*y.
		XXt=np.dot(self.X, self.X.T)
		lambdaI=np.identity(self.X.T.shape[1], float)
		inverse=linalg.inv(XXt+(lambdaI*self.Lambda))
		self.w=np.dot(inverse,np.dot(self.X, self.Y))
		# for i in range(0,len(self.w)-1):
		# 	self.weightMatrix[i][0]=self.w[i]

	def predict(self, Xnew, trainedWeight):
		#used for predicting yHat.
		#regression: yHat=transpose(X)*weights.
		yHat=np.dot(Xnew.T, trainedWeight)
		return yHat

	def accuracy(self, yHat):
		fitVar=0
		regulVar=0
		#minimization formula with fit and regularization.
		#formula: min(w)  Sum((y-yHat)**2)+lambda*sqrt(sum(w**2)).
		#objective is to minimize minimizationVal.
		for i in range(0,len(yHat)-1): 
			fitVar+=np.square((self.Y[i]-yHat[i]))
		for i in range(0,len(self.w)-1):
			regulVar+=np.square(self.w[i])
		minimizationVal=fitVar+self.Lambda*(np.sqrt(regulVar))
		return minimizationVal

def readFile(file, numline, numfeats):
	fileObj=open(file, 'r')
	#input file data will be split into Y and X.
	X=np.zeros([numline, numfeats+1]) 
	Y=np.zeros(numline)
	lineObj=fileObj.readline()
	i=0
	#data input lines are split into field entries.
	while lineObj:
		lineObj=lineObj.strip() 
		entries=lineObj.split()
		Y[i]=int(entries[0])
		X[i][0]=1
		for j in entries[1:]:
			featIdent=j.split(':')
			X[i][int(featIdent[0])]=1
		i+=1
		lineObj=fileObj.readline()
	#X is returned with each col. being an entry.
	return (X.T, Y)

#---Main Call Procedure--- 
for i in range(1, 100, 1):
	i*=0.01

	#Training set used to get weight.
	X, Y=readFile('a7a.train', 16100, 123)
	trainClassifier=reg(X, Y, 123, i)
	trainClassifier.train()

	#Dev set used to find best lambda.
	Xnew, Ynew=readFile('a7a.dev', 8000, 123)
	devSet=reg(Xnew, Ynew, 123, trainClassifier.Lambda)
	devSet.w=trainClassifier.w
	yHat=devSet.predict(Xnew,devSet.w)
	minVal=devSet.accuracy(yHat)
	print "Lambda: ",i , "Score: " ,minVal

	#selection of the minimum score.
	if ((i==0.01) or (minVal<minimum)):
		minimum=minVal
		minLambda=i
print "For min. score, lambda of: ",minLambda , "Score: " ,minimum

#Test set used to find prediction with the calculated weights and selected lambda.
Xtest, Ytest=readFile('a7a.test', 8461, 123)
testSet=reg(Xtest, Ytest, 123, minLambda)
testSet.w=trainClassifier.w
yHat_test=testSet.predict(Xtest,testSet.w)
minVal_test=testSet.accuracy(yHat_test)
print "Test Set Lambda: ",minLambda , "Test Set Score: " ,minVal_test
print "Test Set yHat prediction: " ,yHat_test


