#Nikita Miroshnichenko
#CSC 246, Professor Gildea
#27554869

import numpy as np 
from numpy import linalg
import sys
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

#HMM wtih Gaussians.
class hmm(object):

	#Initialization of X matrix, k (gaussian number), covariance matrix, alpha, beta, gamma, xi, A and B matrix.
	#Mu values for each k gaussian are taken to be random.
	#The matrix A is initialized to random numbers between 0 and 1 for each [i][j] cell.
	#The matrix B is initialized to Gaussian calculations for each [t][i] cell.

	#Tied Covariance matrix is used for each Gaussian.
	def __init__(self, X, k, row_count):
		self.limit=(10**-12)
		self.X=X
		self.k=k
		self.row_count=row_count

		#Alpha initialization.
		self.alpha=np.zeros([len(self.X[:]),self.k])
		self.alpha[0][0]=1

		#Beta initialization.
		self.beta=np.zeros([len(self.X[:]),self.k])
		self.beta[len(self.X[:])-1][:]=1

		#Gamma and Xi initilization.
		self.gamma=np.zeros([len(self.X[:]),self.k])
		self.xi=np.zeros([len(self.X[:]),self.k,self.k])

		#A(i,j) matrix initialization.
		self.A=np.zeros([self.k,self.k])
		for i in range(0,self.k):
			for j in range(0,self.k):
				self.A[i][j]=round(random.uniform(0.1, 1.0), 10)

		#Mean initialization for each Gaussian.
		self.mu_rand=np.zeros([k,2])
		self.mu_rand_temp=random.sample(zip(X[:,0],X[:,1]), k)
		for i in range(0,k):
			self.mu_rand[i][0]=self.mu_rand_temp[i][0]
			self.mu_rand[i][1]=self.mu_rand_temp[i][1]

		#B(t,i) matrix initialization to Gaussian probabilities.
		self.B=np.zeros([len(self.X[:]),self.k])
		self.covar_tied_matrix=np.cov(np.transpose(self.X))

		covar_matrix=self.covar_tied_matrix
		temp_vec_0=np.zeros([1,2])
		temp_vec_1=np.zeros([1,2])
		for t in range(0,self.row_count):
			for i in range(0,self.k):
				temp_val_0=self.X[t][0]-self.mu_rand[i][0]
				temp_val_1=self.X[t][1]-self.mu_rand[i][1]
				temp_vec_0[0][0]=temp_val_0
				temp_vec_0[0][1]=temp_val_1

				inverse_covar_mtx=np.linalg.inv(covar_matrix)
				term_val=np.dot(np.dot(temp_vec_0,inverse_covar_mtx),np.transpose(temp_vec_0))
				nom_val=(np.exp((-0.5)*(float(term_val)))/(math.pi*2*(np.linalg.det(covar_matrix))**(0.5)))

				self.B[t][i]=float(nom_val)

	#E-step procedure used for calculating alpha, beta, P(X) (the normalization constant), xi and gamma.
	def e_step(self):
		#Alpha calculation procedure.
		#Alpha initialization: alpha(0,Start)=1, all other alphas=0. Initialization done in __init__ procedure.
		#Calculation: alpha(t,i) += alpha(t-1,j) * A(i,j) * B(t,i).
		for t in range(0,len(self.X[:])):
			for i in range(0,self.k):
				for j in range(0,self.k):
					if(t==0 and i==0):
						continue;
					else:
						self.alpha[t][i]+=self.alpha[t-1][j]*self.A[i][j]*self.B[t][i]
						if (self.alpha[t][i]<= self.limit):
							self.alpha[t][i]=self.limit
		#Beta calculation procedure.
		#Beta initialization: alpha(N,i)=1, all other betas=0. Initialization done in __init__ procedure.
		#Calculation: beta(t,i) += beta(t+1,j) * A(j,i) * B(t,j).
		for t in range(len(self.X[:]),-1,-1):
			for i in range(0,self.k):
				for j in range(0,self.k):
					if(t<(len(self.X[:])-1) and (i>0)):
						self.beta[t][i]+=self.beta[t+1][i]*self.A[i][j]*self.B[t][i]
						if (self.beta[t][i]<= self.limit):
							self.beta[t][i]=self.limit
		#P(X) calculation procedure.
		#Following 13.42 formula from textbook, normalization constant is P(X).
		#Calculation: P(X) = sum(alpha(Z(n))).
		prob_value=0
		for t in range(0,len(self.X[:])):
			for i in range(0,self.k):
				prob_value+=self.alpha[t][i]*float(self.beta[t][i])
		#Xi calulcation procedure.
		#Calculation: (1/P(X)) * alpha(t-1,j) * P(z(t)=i | z(t-1)=j) * P(x=t | z(t)=i) * beta(t,i).  
		for t in range(0,len(self.X[:])):
			for i in range(0,self.k):
				for j in range(0,self.k):
					self.xi[t][i][j]=float(1/prob_value)*self.alpha[t-1][j]*self.A[i][j]*self.B[t][i]*self.beta[t][i]
		#Gamma calculation procedure.
		#Calculation: (1/P(X)) * alpha(t,i) * beta(t,i)
		for t in range(0,len(self.X[:])):
			for i in range(0,self.k):
				self.gamma[t][i]=float(1/prob_value)*self.alpha[t][j]*self.beta[t][i]

	#M-step procedure used for calculating new A(i,j) and B(t,i).
	def m_step(self):

		#B(t,i) calculation procedure.
		#Calculation: B(t,i) = ec(i,t) / ec(i)
		#ec(i,t) = gamma(t,i)
		#ec(i) = sum(gamma(t,i))
		ec_i=0
		for t in range(0,len(self.X[:])):
			for i in range(0,self.k):
				ec_i+=self.gamma[t][i]

		for t in range(0,len(self.X[:])):
			for i in range(0,self.k):
				self.B[t][i]=self.gamma[t][i]/float(ec_i)

		#A(i,j) calculation procedure.
		#Calculation: A(i,j) = ec(i,j) / ec(j)
		#ec(i,j) = sum(xi(t,i,j))
		#ec(j) = sum(gamma(t,j))
		for i in range(0,self.k):
			for j in range(0,self.k):
				ec_ij=0
				ec_j=0
				for t in range(0,len(self.X[:])):
					ec_ij+=self.xi[t][i][j]
					ec_j+=self.gamma[t][j]
				self.A[i][j]=ec_ij/float(ec_j)
		return self.alpha, self.beta, self.A, self.B, self.xi, self.gamma

	#Log Likelihood procedure used for calculating ln(p(X|mu,sigma) using 9.28 from textbook.
	#For each Gaussian k, mu(mean) and sigma(covariance matrix) are calculated as in 13.20 and 13.21 in the textbook.
	#The calculation uses (x(t)-mu(k))T * (x(t)-mu(k))) instead of (x(t)-mu(k)) * (x(t)-mu(k)))T which is the notation in the textbook,
	#because, the vectors are stored vertically not horizontally.
	def likelihood(self):
		#Mu calculation procedure.
		#mu(k) = (sum (gamma(t,i)) * x(t)) / (sum (gamma(t,i)))
		for i in range(0,self.k):
			num_val_0=0
			num_val_1=0
			denom_val=0
			for t in range(0,len(self.X[:])):
				num_val_0+=self.gamma[t][i]*self.X[t][0]
				num_val_1+=self.gamma[t][i]*self.X[t][1]
				denom_val+=self.gamma[t][i]
			self.mu_rand[i][0]=num_val_0/float(denom_val)
			self.mu_rand[i][1]=num_val_0/float(denom_val)
		#Covariance calculation procedure.
		#sigma(k) =  (sum (gamma(t,i) * (x(t)-mu(k))T * (x(t)-mu(k))) / (sum (gamma(t,i)))
		for i in range(0,self.k):
			num_val=0
			denom_val=0
			for t in range(0,len(self.X[:])):
				temp_vec=np.zeros([1,2])
				temp_vec[0][0]=self.X[t][0]-self.mu_rand[i][0]
				temp_vec[0][1]=self.X[t][1]-self.mu_rand[i][1]

				num_val+=self.gamma[t][i]*np.dot(np.transpose(temp_vec),temp_vec)
				denom_val+=float(self.gamma[t][i])
			self.covar_tied_matrix=num_val/denom_val
		#log likelihood calculation procedure.
		#Calculation: ln(p(X|mu,sigma) = (from 1 to N sum (ln(from 1 to k sum (sum_val_0))) )
		#sum_val_0 = (from 1 to k sum (N(X(n)|mu(k),sigma(k))) ) 
		#N = 900 as train files contains 900 points. k = number of Gaussians used.
		sum_val=0
		for i in range(0,len(self.X[:])):
			sum_val_0=0
			for k in range(0,self.k):
				temp_vec_0=np.zeros([1,2])

				covar_matrix=self.covar_tied_matrix
			
				temp2_val_0=self.X[i][0]-self.mu_rand[k][0]
				temp2_val_1=self.X[i][1]-self.mu_rand[k][1]
				temp_vec_0[0][0]=temp2_val_0
				temp_vec_0[0][1]=temp2_val_1

				inverse_covar_mtx=np.linalg.inv(covar_matrix)

				nom_val=np.exp((-0.5)*np.dot(np.dot(temp_vec_0,inverse_covar_mtx),np.transpose(temp_vec_0)))
				denom_val=float(math.pi*2*(np.linalg.det(covar_matrix))**(0.5))
				sum_val_0+=(nom_val/denom_val)
			if (sum_val_0<self.limit):
				sum_val_0=self.limit
			sum_val+=math.log(sum_val_0)
		return sum_val, self.mu_rand, self.covar_tied_matrix

def readFile(file, numline):
	fileObj=open(file, 'r')
	#Input file data will be split into points of format [x_value, y_value].
	X=np.zeros([numline, 2])
	lineObj=fileObj.readline()
	i=0
	while lineObj:
		lineObj=lineObj.strip()
		entries=lineObj.split()
		X[i][0]=entries[0]
		X[i][1]=entries[1]
		i+=1
		lineObj=fileObj.readline()
	#X matrix is returned with each row being a point.
	return X

#------------Main Call Procedure------------#
if len(sys.argv)>1:
	gauss_num=int(sys.argv[1])
else:
	gauss_num=5
covar_var='tied'
print "Running with paramters:", covar_var, "covar. matrices, and", gauss_num, "mixtures."

#Training set is used on the HMM procedure.
X=readFile('train.txt', 900)
trainClassifier=hmm(X,gauss_num,900)

#Training set initializations.
#Initialized arrays used for graphing.
iteration_array=[]
likelihood_array=[]
convrg_alpha=0
convrg_beta=0
convrg_A=0
convrg_B=0
convrg_xi=0
convrg_gamma=0
convrg_mu=0
convrg_covar=0

compare_val_0=0
count=0
print "Likelihood Values for Train Set:"
while True:
	trainClassifier.e_step()
	convrg_alpha, convrg_beta, convrg_A, convrg_B, convrg_xi, convrg_gamma=trainClassifier.m_step()
	count+=1
	compare_val_1, convrg_mu, convrg_covar=trainClassifier.likelihood()
	print "new likelihood:", compare_val_1, ", old:", compare_val_0, " iteration:",count
	temp_val=abs(abs(compare_val_1)-abs(compare_val_0))

	likelihood_array.append(compare_val_1)
	iteration_array.append(count)

	if (temp_val<=0.00005):
		print "Convergent Train Set Log Likelihood Value: ", compare_val_1
		fig=plt.figure()
		plt.xlabel("Iteration Count")
		plt.ylabel("Log Likelihood")
		plt.title("Train Set, Gauss Num: %s, Covar Matrix: %s"%(gauss_num,covar_var))
		plt.plot(iteration_array, likelihood_array, 'ro')
		plt.show()

		break
	else:
		compare_val_0=compare_val_1

#Dev set is used on the HMM procedure.
X_dev=readFile('dev.txt', 100)
trainClassifier_dev=hmm(X,gauss_num,100)
trainClassifier_dev.mu_rand=convrg_mu
trainClassifier_dev.covar_tied_matrix=convrg_covar
# trainClassifier_dev.alpha=convrg_alpha
# trainClassifier_dev.beta=convrg_beta
# trainClassifier_dev.A=convrg_A
# trainClassifier_dev.B=convrg_B
# trainClassifier_dev.xi=convrg_xi
# trainClassifier_dev.gamma=convrg_gamma

#Dev set initializations.
#Initialized arrays used for graphing.
iteration_array_dev=[]
likelihood_array_dev=[]
compare_val_0=0
count=0
print "----------"
print "Likelihood Values for Dev Set:"
while True:
	trainClassifier_dev.e_step()
	trainClassifier_dev.m_step()
	count+=1
	compare_val_1, temp_mu, temp_covar=trainClassifier_dev.likelihood()
	print "new dev likelihood:", compare_val_1, ", old:", compare_val_0, " iteration:",count
	temp_val=abs(abs(compare_val_1)-abs(compare_val_0))

	likelihood_array_dev.append(compare_val_1)
	iteration_array_dev.append(count)

	if (temp_val<=0.00005):
		print "Convergent Dev Set Log Likelihood Value: ", compare_val_1
		fig=plt.figure()
		plt.xlabel("Iteration Count")
		plt.ylabel("Log Likelihood")
		plt.title("Dev Set, Gauss Num: %s, Covar Matrix: %s"%(gauss_num,covar_var))
		plt.plot(iteration_array_dev, likelihood_array_dev, 'bo')
		plt.show()

		break
	else:
		compare_val_0=compare_val_1





