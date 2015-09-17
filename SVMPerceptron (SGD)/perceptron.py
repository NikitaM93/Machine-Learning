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

#Perceptron with Stochastic Gradient Descent.
class perceptron(object):

	#Initialization of X matrix, Y, Prediction (Y_hat) and W vectors.
	def __init__(self, X, Y, numfeats, set_string):
		self.Set=set_string
		self.X=X
		self.Y=Y
		self.W=np.ones([numfeats+1,1])
		self.numfeats=numfeats
		self.Y_hat=[0]*(len(self.Y))

	#Train procedure for training weight (W) vector.
	#Eta value (temp_val) is set to decrease with every interation.
	#Weight vector is adjusted by looking at each x_i entry. 
	#Each x_i entry is a single column in the X matrix.
	#Each of these x_i columns are referenced as x_i_feats columns.
	def train(self):
		#Training is repeated 10 times to improve on accuracy for the weight vector.
		#Each iteration count affects the value of the Eta value (temp_val).
		#From lecture, the following is the expression for Eta: 
		#Eta(k)=(1/k), where (k) is the iteration count.
		for iteration in range(1,11):
			func_gradient=0.0
			temp_val=(float(1)/(iteration))

			for x_i in range(1,len(self.X[0])):

				x_i_feats=np.zeros([self.numfeats+1,1])
				for feat in range(1,self.numfeats+1):
					x_i_feats[feat]=(self.X[feat][x_i])

				#From lecture, the update rule for the weight vector is as follows:
				#if Y(k)*X_column(k)<0 then the following updates,
				#gradient_f(w) = gradient_f(w)-(Y(k)*X_column(k)).
				#If this condition does not hold then nothing is added to gradient_f(w).
				cond_value=(self.Y[x_i])*(np.dot(np.transpose(self.W),(x_i_feats)))
				if (cond_value<0):
					func_gradient-=np.dot((self.Y[x_i]),x_i_feats)
			self.W-=temp_val*(func_gradient)/(self.numfeats)

	#Predict procedure for predicting Y_hat, using the trained Weight vector.
	#From lecture, the following is the equation for Y_hat:
	#Y_hat=(transpose(X)*W).
	def predict(self):
		self.Y_hat=np.dot(self.X.T, self.W)
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
	def accuracy(self):
		#From lecture, the MSE formula is used:
		#Sum((y-yHat)**2).
		#The Main Call Procedure later finds the accuracy score.
		fit_var=0
		for i in range(0,len(self.Y_hat)):
			fit_var+=np.square((self.Y[i]-self.Y_hat[i]))
		print fit_var, " Accuracy score in set: ", self.Set
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

#Conditional statements for identifying test file and test file line count.
#If the command line does not read any argument then the procedure uses the a7a.test as the test file.
#If the comman line does read an argument then the argument is saved to be the test file's name, which is later read.
test_len=0
test_name=""
if len(sys.argv)>1:
    test_len=file_len(sys.argv[1])
    test_name=sys.argv[1]
else:
    test_name="a7a.test"
    test_len=8461

#Training set used for training Perceptron weight vector.
X, Y=readFile('a7a.train', 16100, feature_num)
trainClassifier=perceptron(X, Y, feature_num,'a7a.train')
trainClassifier.train()

#Dev set used with the trained weight vector.
#Accuracy score and percentage accuracy is displayed.
X_dev, Y_dev=readFile('a7a.dev', 8000, feature_num)
devSet=perceptron(X_dev, Y_dev, feature_num,'a7a.dev')
devSet.W=trainClassifier.W
devSet.predict()
acc_score=devSet.accuracy()
acc_percent=devSet.accuracy_percent()

#Test set used with the trained weight vector, showing the respective accuracy.
#Predicted Y_hat values are also printed out to command line.
X_test, Y_test=readFile(test_name, test_len, feature_num)
testSet=perceptron(X_test, Y_test, feature_num,test_name)
testSet.W=trainClassifier.W
testSet.predict()

print "------Predictions for Test Entries: "
for i in range(0,len(testSet.Y_hat)):
	print testSet.Y_hat[i], " predicted. Actual Y value is: ",testSet.Y[i]
print "------Test Set Results for File:",test_name,"------"
test_score=testSet.accuracy()
test_percent=(testSet.accuracy_percent())*100
