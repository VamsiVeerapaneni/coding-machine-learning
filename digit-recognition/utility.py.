
# Import the necessary Packages
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
	# Load the train and test data
	df=pd.read_csv('../data/train(1).csv')
	test=pd.read_csv('../data/test.csv')
	# Obtain the train and test set
	y=df.ix[:,0]
	x=df.ix[:,1:785]
	y_train=np.array(y)
	train=np.array(x)
	test=np.array(test)
	return train,y_train,test

def perform_PCA(train,test,axis):
	# Build and Transform using PCA
	pca=PCA(n_components=axis,whiten='True')
	pca.fit(train)
	x_train=pca.transform(train)
	test=pca.transform(test)
	return x_train,test

