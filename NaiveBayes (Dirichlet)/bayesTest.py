#Nikita Miroshnichenko
#CSC 246, Professor Gildea
#27554869

import numpy as np 
from numpy import linalg
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import matplotlib.pyplot as plt

class naive_bayes(object):
	def __init__(self, X, Y, numfeats, Alpha):
		#procedure matrices initialization.
		self.X=X
		self.Y=Y
		self.Alpha=Alpha
		self.feat_prob_plus=[0]*(feature_num+1)
		self.feat_prob_minus=[0]*(feature_num+1)
		self.y_plus_prob=0
		self.y_minus_prob=0
		self.set_plus_prob=[0]*(len(self.X[0])-1)
		self.set_minus_prob=[0]*(len(self.X[0])-1)
		self.y_hat=[0]*(len(self.Y))

	def train(self):
		y_plus_num=0
		y_minus_num=0
		feat_y_plus=[0]*(feature_num+1)
		feat_y_minus=[0]*(feature_num+1)
		#count for number of y = +1 and y = -1, followed by P(y = -1) and p(y = +1) calculations.
		for y in range(0,len(self.Y)):
			if (self.Y[y]==(+1)):
				y_plus_num+=1
			if (self.Y[y]==(-1)):
				y_minus_num+=1
		self.y_plus_prob=float(y_plus_num)/len(self.Y)
		self.y_minus_prob=float(y_minus_num)/len(self.Y)
		#Naive Bayes (with Dirichlet smoothing) probability calculation.
		#procedure used for calculating probability of y = +1 and y = -1 for each feature.
		for i in range(1,feature_num+1):
			for x in range(0,len(self.X[0])-1):
				if((self.X[i][x]==1) and (self.Y[x]==(+1))):
					feat_y_plus[i]+=1
				if((self.X[i][x]==1) and (self.Y[x]==(-1))):
					feat_y_minus[i]+=1
			#Naive Bayes (with Dirichlet smoothing) probability for y = +1.
			numerator_pos=np.add(float(feat_y_plus[i]),self.Alpha)
			denominator_pos=np.add(y_plus_num,np.multiply(feature_num+1,self.Alpha))
			self.feat_prob_plus[i]=np.divide(numerator_pos,denominator_pos)
			#Naive Bayes (with Dirichlet smoothing) probability for y = -1.
			numerator_neg=np.add(float(feat_y_minus[i]),self.Alpha)
			denominator_neg=np.add(y_minus_num,np.multiply(feature_num+1,self.Alpha))
			self.feat_prob_minus[i]=np.divide(numerator_neg,denominator_neg)

	def predict(self):
		#procedure used for calculating Naive Bayesian probability of y = +1 and y = -1 for each x_i col.
		#the Naive Bayesian probability uses the independence assumption.
		for x_i in range(0,len(self.X[0])-1):
			self.set_plus_prob[x_i]=self.y_plus_prob
			self.set_minus_prob[x_i]=self.y_minus_prob
			for i in range(1,feature_num+1):
				if (self.X[i][x_i]==1):
					self.set_plus_prob[x_i]*=self.feat_prob_plus[i]
					self.set_minus_prob[x_i]*=self.feat_prob_minus[i]

	def accuracy(self):
		#procedure used for calculating accuracy score by using MSE formula.
		#formula: Sum((y-yHat)**2).
		#main call procedure later finds the minimum accuracy score.
		for i in range(0,len(self.set_plus_prob)-1):
			if (self.set_plus_prob[i]>=self.set_minus_prob[i]):
				self.y_hat[i]=(+1)
			else:
				self.y_hat[i]=(-1)
		fit_var=0
		for i in range(0,len(self.y_hat)-1):
			fit_var+=np.square((self.Y[i]-self.y_hat[i]))
		return fit_var

	def accuracy_percent(self):
		#procedure used for calculating percentage accuracy of the prediction.
		#formula: (number of correct predictions)/(total number of y's).
		for i in range(0,len(self.set_plus_prob)-1):
			if (self.set_plus_prob[i]>=self.set_minus_prob[i]):
				self.y_hat[i]=(+1)
			else:
				self.y_hat[i]=(-1)
		num_correct=0
		for i in range(0,len(self.y_hat)-1):
			if (self.y_hat[i]==self.Y[i]):
				num_correct+=1
		percent=float(num_correct)/len(self.Y)
		return percent

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
feature_num=123
minimum=""
alphas=[]
scores=[]

for i in range(1,50,1):
	alpha=i*0.2
	#Training set used for training Naive Bayesian probabilities with Dirichlet Prior.
	X, Y=readFile('a7a.train', 16100, feature_num)
	trainClassifier=naive_bayes(X, Y, feature_num, alpha)
	trainClassifier.train()
	#Dev set used for finding best alpha, by comparing accuracies.
	X_dev, Y_dev=readFile('a7a.dev', 8000, feature_num)
	devSet=naive_bayes(X_dev, Y_dev, feature_num, alpha)
	devSet.feat_prob_plus=trainClassifier.feat_prob_plus
	devSet.feat_prob_minus=trainClassifier.feat_prob_minus
	devSet.y_plus_prob=trainClassifier.y_plus_prob
	devSet.y_minus_prob=trainClassifier.y_minus_prob
	devSet.predict()
	acc_score=devSet.accuracy()
	acc_percent=devSet.accuracy_percent()
	print "Dev Accuracy: ",acc_score, ", Percent: ", acc_percent, ", Alpha: ", alpha
	alphas.append(alpha)
	scores.append(acc_score)

	#calculation of minimimu MSE score from alpha iteration.
	if ((i==0.02) or ((acc_score<minimum) and ~isinstance(minimum, str))):
		minimum=acc_score
		minAlpha=alpha
		bestPercent=acc_percent
print "Dev: For min. MSE score, alpha of: ",minAlpha ,", Percent: ",bestPercent, ", Score: " ,minimum

#procedure for plot of accuracy scores against alpha for Dev set.
# fig=plt.figure()
# fig.suptitle('Classification Error against Alpha', fontsize=18)
# plt.ylabel('MSE Score')
# plt.xlabel('Alpha')
# plt.plot(alphas, scores, 'ro')
# plt.show()

#Test set used with the selected alpha to show the respective accuracy.
X_test, Y_test=readFile('a7a.test', 8461, feature_num)
testSet=naive_bayes(X_test, Y_test, feature_num, minAlpha)
testSet.feat_prob_plus=trainClassifier.feat_prob_plus
testSet.feat_prob_minus=trainClassifier.feat_prob_minus
testSet.y_plus_prob=trainClassifier.y_plus_prob
testSet.y_minus_prob=trainClassifier.y_minus_prob
testSet.predict()
test_score=testSet.accuracy()
test_percent=testSet.accuracy_percent()
print "Test Accuracy: ",test_score, ", Percent: ", test_percent, ", Alpha: ", minAlpha
