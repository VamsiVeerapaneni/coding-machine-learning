from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

visualization=True

def create_data():
	x,y=make_circles(n_samples=800,noise=.07,factor=.4)
	if visualization:
		plt.scatter(x[:,0],x[:,1],c=y+1)
		plt.show()
	return x,y

def project_data(data):
	x=data[:,0]
	y=data[:,1]
 	z=x**2+y**2
	return x,y,z

data,label=create_data()
x,y,z=project_data(data)
if visualization:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x[label==0],y[label==0],z[label==0],c='r')
	ax.scatter(x[label==1],y[label==1],z[label==1],c='b')
	plt.show()
