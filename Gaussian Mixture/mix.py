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

#EM for Mixture of Gaussians.
class mix(object):

	#Initialization of X matrix, k (gaussian number), covariance matrices, lambda and mu values.
	#Mu values for each k gaussian are taken to be random.
	#The matrix for P(Z=k|X(n)) values is initialized as self.z.

	#If command line reads 'tied' then one covariance matrix (covaraince matrix of the whole X matrix)
	#will be used for each gaussian's covariance matrix.
	#If command line reads 'separate' then a different covaraince matrix will be generated for each gaussian.
	def __init__(self, X, k, row_count, covar_var):
		self.X=X
		self.k=k
		self.row_count=row_count
		self.covar_var=covar_var
		self.covar_dict=dict()
		self.covar_tied_matrix=0
		self.mu_rand=np.zeros([k,2])

		self.mu_rand_temp=random.sample(zip(X[:,0],X[:,1]), k)
		for i in range(0,k):
			self.mu_rand[i][0]=self.mu_rand_temp[i][0]
			self.mu_rand[i][1]=self.mu_rand_temp[i][1]

		self.lam=np.zeros([k])
		for i in range(0,k):
			self.lam[i]=1/float(k)

		self.z=np.zeros([self.row_count,k])

		#Identifying whether tied or separate covaraince matrices will be used in EM procedure.
		#If tied, then one covariance matrix is created.
		#If separate, then different covariance matrices are made for each gaussian.
		if (self.covar_var=='tied'):
			self.covar_tied_matrix=np.cov(np.transpose(self.X))
		elif (self.covar_var=='separate'):
			for k in range(0,k):
				sum_val_xx=0
				sum_val_xy=0
				sum_val_yy=0
				for i in range(0,self.row_count):
					temp_val_0=self.X[i][0]-self.mu_rand_temp[k][0]
					temp_val_1=self.X[i][0]-self.mu_rand_temp[k][0]
					sum_val_xx+=(temp_val_0*temp_val_1)

					temp_val_0=self.X[i][0]-self.mu_rand_temp[k][0]
					temp_val_1=self.X[i][1]-self.mu_rand_temp[k][1]
					sum_val_xy+=(temp_val_0*temp_val_1)

					temp_val_0=self.X[i][1]-self.mu_rand_temp[k][1]
					temp_val_1=self.X[i][1]-self.mu_rand_temp[k][1]
					sum_val_yy+=(temp_val_0*temp_val_1)

				temp_array=np.zeros([2,2])
				temp_array[0][0]=float(sum_val_xx)/self.row_count
				temp_array[0][1]=float(sum_val_xy)/self.row_count
				temp_array[1][0]=float(sum_val_xy)/self.row_count
				temp_array[1][1]=float(sum_val_yy)/self.row_count
				self.covar_dict[k]=temp_array

	#E-step procedure used for calculating P(Z=k|X(n)).
	#Calculations is as follows:
	#P(Z=k|X(n)) = lambda[k] * N(X(n)|mu(k),sigma(k)) / ( from 1 to k sum(lambda[k] * N(X(n)|mu(k),sigma(k)) ) )
	#N(X(n)|mu(k),sigma(k)) = exp( (-0.5) * (X(n)-mu(k))T * inv.sigma(k) * (X(n)-mu(k)) ) / ( ((2*pi)^(D/2)) * (det.sigma(k))^0.5 ) 
	#P(Z=k|X(n)) is set to be a (k,row_count) matrix.
	#Each entry in this probability matrix shows probability that a point lies within a certain gaussian.
	def e_step(self):

		#Formula value initializations.
		temp_vec_0=np.zeros([1,2])
		temp_vec_1=np.zeros([1,2])

		#Formula value calculations.
		for i in range(0,self.row_count):
			
			#Calculation of the denominator of the P(Z=k|X(n)) expression.
			denom_val=0
			#Summation procedure for all k Gaussians.
			for k in range(0,self.k):
				if (self.covar_var=='tied'):
					covar_matrix=self.covar_tied_matrix
				else:
					covar_matrix=self.covar_dict[k]

				temp2_val_0=self.X[i][0]-self.mu_rand[k][0]
				temp2_val_1=self.X[i][1]-self.mu_rand[k][1]
				temp_vec_1[0][0]=temp2_val_0
				temp_vec_1[0][1]=temp2_val_1

				inverse_covar_mtx=np.linalg.inv(covar_matrix)

				term2_val=np.dot(np.dot(temp_vec_1,inverse_covar_mtx),np.transpose(temp_vec_1))
				denom_val+=float(self.lam[k])*(np.exp((-0.5)*(term2_val))/(math.pi*2*(np.linalg.det(covar_matrix))**(0.5)))

			#Calculation of the numerator of the P(Z=k|X(n)) expression.
			for j in range(0,self.k):
				if (self.covar_var=='tied'):
					covar_matrix=self.covar_tied_matrix
				else:
					covar_matrix=self.covar_dict[j]
				
				temp_val_0=self.X[i][0]-self.mu_rand[j][0]
				temp_val_1=self.X[i][1]-self.mu_rand[j][1]
				temp_vec_0[0][0]=temp_val_0
				temp_vec_0[0][1]=temp_val_1

				inverse_covar_mtx=np.linalg.inv(covar_matrix)
				term_val=np.dot(np.dot(temp_vec_0,inverse_covar_mtx),np.transpose(temp_vec_0))
				nom_val=(self.lam[j]*np.exp((-0.5)*(term_val)))/(math.pi*2*(np.linalg.det(covar_matrix))**(0.5))
				
				self.z[i][j]=nom_val/denom_val

	#M-step procedure used for calculating new mean (mu(k)), new covariance matrix (sigma(k)) and new lambda (lam(k)).
	#Calculations is as follows:
	#mu(k) = ( from 1 to N sum(P(Z=k|X(n)) * X(n) ) ) / ( from 1 to N sum(P(Z=k|X(n)) )
	#sigma(k) = ( from 1 to N sum(P(Z=k|X(n)) * (X(n)-mu(k)) * (X(n)-mu(k))T) ) / ( from 1 to N sum(P(Z=k|X(n)) )
	#lambda(k) = ( from 1 to N sum(P(Z=k|X(n))) ) / N
	#The procedure's mu(k), sigma(k), and lam(k) values get updated to this newly calculated values.
	#In the case of 'tied' covariance matrices, the sigma (covariance matrix) that is used
	#for all of the gaussians is not updated.
	def m_step(self):

		for i in range(0,self.k):
			#Formula value initializations.
			lam_nom_val=0
			mu_nom_val=np.zeros([1,2])
			mu_denom_val=0
			covar_nom_val=np.zeros([2,2])
			covar_denom_val=0

			temp_vec=np.zeros([1,2])

			#Formula value calculations.
			for j in range(0,self.row_count):
				lam_nom_val+=self.z[j][i]
				temp_vec[0][0]=self.X[j][0]
				temp_vec[0][1]=self.X[j][1]

				mu_nom_val+=((self.z[j][i])*temp_vec)
				mu_denom_val+=(self.z[j][i])

				temp_val_0=self.X[j][0]-self.mu_rand[i][0]
				temp_val_1=self.X[j][1]-self.mu_rand[i][1]
				temp_vec[0][0]=temp_val_0
				temp_vec[0][1]=temp_val_1

				covar_nom_val+=(self.z[j][i])*np.dot(np.transpose(temp_vec),temp_vec)
				covar_denom_val+=(self.z[j][i])

			self.lam[i]=(float(lam_nom_val)/self.row_count)
			self.mu_rand[i][0]=mu_nom_val[0][0]/float(mu_denom_val)
			self.mu_rand[i][1]=mu_nom_val[0][1]/float(mu_denom_val)

			if (self.covar_var=='separate'):
				self.covar_dict[i]=(covar_nom_val)/float(covar_denom_val)

		if (self.covar_var=='tied'):
			return self.mu_rand, self.lam, self.covar_tied_matrix
		else:
			return self.mu_rand, self.lam, self.covar_dict

	#Log Likelihood procedure used for calculating ln(p(X|mu,sigma,lambda).
	#The current log likelihood value is compared to the previous one in the main call method.
	#The main call method is located at the bottom of this file.
	#Calculations is as follows:
	#ln(p(X|mu,sigma,lambda) = ( from 1 to N sum(ln( from 1 to k sum(sum_val_0) )) )
	#sum_val_0 = ( from 1 to k sum(lambda[k] * N(X(n)|mu(k),sigma(k)) ) ) 
	#The procedure accounts for 'tied' and 'separate' covariance matrices, as specified by command line arguments.
	def likelihood(self):
		#Formula value initialization.
		sum_val=0

		#Formula value calculations.
		for i in range(0,self.row_count):
			sum_val_0=0
			for k in range(0,self.k):
				temp_vec_0=np.zeros([1,2])

				if (self.covar_var=='tied'):
					covar_matrix=self.covar_tied_matrix
				else:
					covar_matrix=self.covar_dict[k]

				temp2_val_0=self.X[i][0]-self.mu_rand[k][0]
				temp2_val_1=self.X[i][1]-self.mu_rand[k][1]
				temp_vec_0[0][0]=temp2_val_0
				temp_vec_0[0][1]=temp2_val_1

				inverse_covar_mtx=np.linalg.inv(covar_matrix)

				nom_val=np.exp((-0.5)*np.dot(np.dot(temp_vec_0,inverse_covar_mtx),np.transpose(temp_vec_0)))
				denom_val=float(math.pi*2*(np.linalg.det(covar_matrix))**(0.5))
				sum_val_0+=(self.lam[k]*nom_val/denom_val)
			sum_val+=math.log(sum_val_0)
		return sum_val

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
if len(sys.argv)>2:
    covar_var=sys.argv[1]
    gauss_num=int(sys.argv[2])
else:
    covar_var='tied'
    gauss_num=5
print "Running with paramters:", covar_var, "covar. matrices, and", gauss_num, "mixtures."

#Training set is used on the EM procedure.
X=readFile('train.txt', 900)
trainClassifier=mix(X,gauss_num,900,covar_var)

#Training set initializations.
#Initialized arrays used for graphing.
iteration_array=[]
likelihood_array=[]
convrg_mu=0
convrg_covar=0
convrg_lam=0

compare_val_0=0
count=0
print "Likelihood Values for Train Set:"
while True:
	trainClassifier.e_step()
	convrg_mu, convrg_lam, convrg_covar=trainClassifier.m_step()
	count+=1
	compare_val_1=trainClassifier.likelihood()
	print "new likelihood:", compare_val_1, ", old:", compare_val_0, " iteration:",count
	temp_val=abs(abs(compare_val_1)-abs(compare_val_0))

	likelihood_array.append(compare_val_1)
	iteration_array.append(count)

	if (temp_val<=0.05):
		print convrg_mu, convrg_lam, convrg_covar
		print "Convergent Train Set Log Likelihood Value: ", compare_val_1
		fig = plt.figure()
		plt.xlabel("Iteration Count")
		plt.ylabel("Log Likelihood")
		plt.title("Train Set, Gauss Num: %s, Covar Matrix: %s"%(gauss_num,covar_var))
		plt.plot(iteration_array, likelihood_array, 'ro')
		plt.show()

		break
	else:
		compare_val_0=compare_val_1

#Dev set is used on the EM procedure.
X_dev=readFile('dev.txt', 100)
trainClassifier_dev=mix(X,gauss_num,100,covar_var)
trainClassifier_dev.mu_rand=convrg_mu
trainClassifier_dev.lam=convrg_lam
trainClassifier_dev.covar_dict=convrg_covar

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
	compare_val_1=trainClassifier_dev.likelihood()
	print "new dev likelihood:", compare_val_1, ", old:", compare_val_0, " iteration:",count
	temp_val=abs(abs(compare_val_1)-abs(compare_val_0))

	likelihood_array_dev.append(compare_val_1)
	iteration_array_dev.append(count)

	if (temp_val<=0.05):
		print "Convergent Dev Set Log Likelihood Value: ", compare_val_1
		fig = plt.figure()
		plt.xlabel("Iteration Count")
		plt.ylabel("Log Likelihood")
		plt.title("Dev Set, Gauss Num: %s, Covar Matrix: %s"%(gauss_num,covar_var))
		plt.plot(iteration_array_dev, likelihood_array_dev, 'ro')
		plt.show()

		break
	else:
		compare_val_0=compare_val_1

