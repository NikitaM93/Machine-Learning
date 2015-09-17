#Aly Grealish
#hw 7 EM
import numpy as np
from numpy import linalg
import sys
import random
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

class EM(object):
	def __init__(self, X_, k_clusters_):
		self.d = dict()
		self.X = X_
		self.k_clusters = k_clusters_
		self.prob_z = np.zeros([len(self.X), k_clusters_])
		self.mylambda = np.zeros([k_clusters_])
		self.mu = np.zeros([k_clusters_, 2])
		for k in range(0, k_clusters_):
			self.d[k] = np.cov(np.transpose(self.X)) #initialize the covariance matrices
		self.mu_val = random.sample(zip(X[:,0],X[:,1]), k_clusters_) #initialize the gaussians
		for x in range(0, k_clusters_):
			self.mu[x][0] = self.mu_val[x][0]
			self.mu[x][1] = self.mu_val[x][1]
		for gauss in range(0, self.k_clusters): #initialize lambda
			self.mylambda[gauss] = float(1)/k_clusters_		

	def estimation(self):
		for n in range(0, len(self.X)):
			denominator = 0
			for y in range(0, self.k_clusters):
				denominator += self.mylambda[y]*(np.exp(-.5 * np.dot(np.dot((self.X[n][:] - self.mu[y][:]), np.linalg.inv(self.d[y])), np.transpose(self.X[n][:] - self.mu[y][:]))))/(math.pi*2*(np.linalg.det(self.d[y]))**(0.5))
			for cluster_num in range(0, self.k_clusters):
				numerator = 0
				one = np.dot((self.X[n][:] - self.mu[cluster_num][:]), np.linalg.inv(self.d[cluster_num]))
				two = np.dot(one, np.transpose(self.X[n][:] - self.mu[cluster_num][:]))
				numerator = np.multiply(self.mylambda[cluster_num], np.exp(-.5 * two))/(2 * math.pi * (np.linalg.det(self.d[cluster_num]))**(0.5))
				self.prob_z[n][cluster_num] = np.divide(numerator, denominator)

	def maximization(self):
		for cluster_num in range(0, self.k_clusters):
			mu_numerator = np.zeros([1, 2])
			mu_denominator = 0
			lambda_numerator = 0
			sigma_numerator = np.zeros([2, 2])
			sigma_denominator = 0
			sigma_val = np.zeros([1,2])
			for n in range(0, len(self.X)):
				mu_numerator += self.prob_z[n][cluster_num] * self.X[n][:]
				mu_denominator += self.prob_z[n][cluster_num]
				lambda_numerator += self.prob_z[n][cluster_num]
				sigma_val[0][0] = self.X[n][0] - self.mu[cluster_num][0]
				sigma_val[0][1] = self.X[n][1] - self.mu[cluster_num][1]
				sigma_numerator += self.prob_z[n][cluster_num] * np.transpose(sigma_val) * sigma_val
				sigma_denominator += self.prob_z[n][cluster_num]
			self.mylambda[cluster_num] = float(lambda_numerator)/len(self.X)
			self.mu[cluster_num][0] = float(mu_numerator[0][0])/mu_denominator
			self.mu[cluster_num][1] = float(mu_numerator[0][1])/mu_denominator
			self.d[cluster_num] = sigma_numerator/float(sigma_denominator)

	def loglikelihood(self):
		summationout = 0
		for n in range(0, len(self.X)):
			numerator = 0 
			denominator = 0
			summationin = 0
			for cluster_num in range(0, self.k_clusters-1):
				numval = np.zeros([1,2])
				numval[0][0] = self.X[n][0] - self.mu[cluster_num][0]
				numval[0][1] = self.X[n][1] - self.mu[cluster_num][1]			
				one = np.dot(numval, np.linalg.inv(self.d[cluster_num]))
				two = np.dot(one, np.transpose(numval))
				numerator = np.exp(-.5 * two)
				denominator = (2 * math.pi) * (np.linalg.det(self.d[cluster_num]))**(0.5)
				summationin += numerator * denominator * self.mylambda[cluster_num]
			summationout += math.log(summationin[0][0])
		return summationout

def readData(file, n_line):
	X = np.zeros([n_line, 2])
	f = open(file, 'r')
	line = f.readline()
	line_no = 0
	while line:
		line = line.strip() #gets rid of white space
		fields = line.split()
		X[line_no][0] = fields[0]
		X[line_no][1] = fields[1]
		line_no +=1
		line = f.readline()
	return X

likelihoodforplot = np.zeros(25)	
count = 0
for i in range(2, 4, 2):
	kclusters = 5
	X = readData('train.txt', 900)
	classifier = EM(X, kclusters)
	convergenceold = 0
	while True:
		classifier.estimation()
		classifier.maximization()
		convergencenew = classifier.loglikelihood()
		if (abs(abs(convergencenew) - abs(convergenceold)) <= 5):
			likelihoodforplot[count] = convergencenew
			count += 1
			break
		else:
			convergenceold = convergencenew

mixnum = range(2, 102, 4)
fig = plt.figure()
plt.ylabel("likelihood")
plt.xlabel("number of mixtures")
plt.plot(mixnum, likelihoodforplot, 'bo')
plt.show()
