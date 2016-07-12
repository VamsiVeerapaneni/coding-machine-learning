from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
x,y=make_circles(n_samples=800,noise=.07,factor=.4)
plt.scatter(x[:,0],x[:,1],c=y+1)
plt.show()
