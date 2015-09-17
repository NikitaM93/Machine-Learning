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
import sys

#SVM with Stochastic Gradient Descent.
class svm(object):

	#Initialization of X matrix, Y, Prediction (Y_hat) and W vectors.
	#Includes initialization of b constant for intercept.
	def __init__(self, X, Y, numfeats, c_value, set_string):
		self.Set=set_string
		self.X=X
		self.Y=Y
		self.C=c_value
		self.B=0.0
		self.W=np.zeros([numfeats+1,1])
		self.numfeats=numfeats
		self.Y_hat=[0]*(len(self.Y))

	#Train procedure for training weight (W) vector.
	#Eta value (temp_Eta) is set to decrease with every interation.
	#Weight vector is adjusted by looking at each x_i entry. 
	#Each x_i entry is a single column in the X matrix.
	#Each of these x_i columns are referenced as x_i_feats columns.
	#This train implementation attempts to maximize the margin.
	def train(self):
		#Training is repeated 10 times to improve on accuracy for the weight vector.
		for iteration in range(0,10):
			for x_i in range(1,len(self.X[0])):
				temp_Eta=(float(1)/(x_i))
				
				x_i_feats=np.zeros([self.numfeats+1,1])
				for feat in range(1,self.numfeats+1):
					x_i_feats[feat]=(self.X[feat][x_i])

				#From lecture, the update rule for the weight vector is as follows:
				#if 1-Y(n)*(transpose(W)*X_column(n)+b)>0 then the following updates,
				#W <- W-(eta)*(W-C*Y_hat*X_column(n)) and b <- b+(C*Y(n)).
				#If this condition does not hold then the following updates,
				#W <- W-(eta)*W
				cond_value=(1-((self.Y[x_i])*(np.dot(np.transpose(self.W),(x_i_feats))+self.B)))
				if (cond_value>0):
					self.B+=(self.C*(self.Y[x_i]))
					self.W-=temp_Eta*(self.W-((self.C*self.Y[x_i])*x_i_feats))
				else:
					self.W-=(temp_Eta*self.W)

	#Predict procedure for predicting Y_hat, using the trained Weight vector and given C value.
	#From lecture, the following is the equation for Y_hat:
	#Y_hat=(transpose(X)*W)+b
	#This equation includes the intercept b. 
	def predict(self):
		self.Y_hat=np.dot(self.X.T, self.W)+self.B
		#From lecture, the sign(x) function is implemented.
		#The 'sign' function is used on Y_Hat values as follows:
		#sign(Y_Hat)=(0) if Y_Hat=0, sign(Y_Hat)=(-1) if Y_Hat<0, sign(Y_Hat)=(1) if Y_Hat>0.
		countpos=0
		countneg=0
		for i in range(0,len(self.Y_hat)):
			if self.Y_hat[i]>0:
				self.Y_hat[i]=1
				countpos+=1
			elif self.Y_hat[i]<0:
				self.Y_hat[i]=(-1)
				countneg+=1

	#Accuracy procedure calculates the MSE accuracy score.
	#The objective is to find the best C value that minimizes this score.
	def accuracy(self):
		#From lecture, the MSE formula is used:
		#Sum((y-yHat)**2).
		#The Main Call Procedure later finds the minimum accuracy score.
		fit_var=0
		for i in range(0,len(self.Y_hat)):
			fit_var+=np.square((self.Y[i]-self.Y_hat[i]))
		print fit_var, " Accuracy score for C value: ", self.C, " in set: ", self.Set
		return fit_var

	#Accuracy_percent procedure outputs the % accuracy of the predictions.
	#The predictions (Y_hat) are compared against the actual values (Y).
	def accuracy_percent(self):
		#The following calculation is used:
		#(number of correct predictions)/(total number of of entries in Y).
		num_correct=0
		for i in range(0,len(self.Y_hat)):
			if (self.Y_hat[i]==self.Y[i]):
				num_correct+=1

		percent=float(num_correct)/len(self.Y)
		print (percent*100), " % Accuracy."
		return percent

#ReadFile procedure reads the input file and creates the Y vector and X matrix.
#Procedure is from the TA help session.
def readFile(file, numline, numfeats):
	fileObj=open(file, 'r')
	#Input file data will be split into Y and X.
	X=np.zeros([numline, numfeats+1]) 
	Y=np.zeros(numline)
	lineObj=fileObj.readline()
	i=0
	#Data input lines are split into field entries.
	while lineObj:
		lineObj=lineObj.strip() 
		entries=lineObj.split()
		Y[i]=int(entries[0])
		X[i][0]=0
		for j in entries[1:]:
			featIdent=j.split(':')
			X[i][int(featIdent[0])]=1
		i+=1
		lineObj=fileObj.readline()
	#X is returned with each column being an independent node entry.
	return (X.T, Y)

#File_len procedure is used to calculate the number of lines in a file.
def file_len(file):
    with open(file) as f:
        for counter, temp in enumerate(f):
            pass
    return counter+1

#------------Main Call Procedure------------#
feature_num=123

#Initialization of minimum MSE score, weight vector and b value for the best C value.
minimum=""
acc_weight=np.zeros([feature_num+1,1])
acc_b=0

#Conditional statements for identifying test file and test file line count.
#If the command line does not read any argument then the procedure uses the a7a.test as the test file.
#If the command line does read an argument then the argument is saved to be the test file's name, which is later read.
test_len=0
test_name=""
if len(sys.argv)>1:
    test_len=file_len(sys.argv[1])
    test_name=sys.argv[1]
else:
    test_name="a7a.test"
    test_len=8461

#Iterating call to test SVM on different values of C.
#For each C value, the loop trains the SVM on the train file and then finds the accuracy in the dev file.
#The C value yielding the best accuracy on the dev file is used on the test file.
for i in range(100,1000,100):
	c_value=i
	#Training set used for training SVM weight vector.
	X, Y=readFile('a7a.train', 16100, feature_num)
	trainClassifier=svm(X, Y, feature_num, c_value,'a7a.train')
	trainClassifier.train()

	#Dev set used for finding best C value, by comparing accuracies.
	X_dev, Y_dev=readFile('a7a.dev', 8000, feature_num)
	devSet=svm(X_dev, Y_dev, feature_num, c_value,'a7a.dev')
	devSet.W=trainClassifier.W
	devSet.B=trainClassifier.B
	devSet.predict()
	acc_score=devSet.accuracy()
	acc_percent=devSet.accuracy_percent()

	#Calculation of minimimu MSE score from the C value iteration.
	if ((i==100) or ((acc_score<minimum) and ~isinstance(minimum, str))):
		minimum=acc_score
		minC=c_value
		acc_b=devSet.B
		acc_weight=devSet.W
		bestPercent=(acc_percent*100)
print "Dev: For min. MSE score, C value of: ",minC ,", Percent : ",bestPercent, "%, Score: " ,minimum

#Test set used with the selected C value to show the respective accuracy.
#Predicted Y_hat values are also printed out to command line.
X_test, Y_test=readFile(test_name, test_len, feature_num)
testSet=svm(X_test, Y_test, feature_num, minC,test_name)
testSet.W=acc_weight
testSet.B=acc_b
testSet.predict()

print "------Predictions for Test Entries: "
for i in range(0,len(testSet.Y_hat)):
	print testSet.Y_hat[i], " predicted. Actual Y value is: ",testSet.Y[i]
print "------Test Set Results for File:",test_name,"------"
test_score=testSet.accuracy()
test_percent=(testSet.accuracy_percent())*100

