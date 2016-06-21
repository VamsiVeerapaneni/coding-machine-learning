from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
SIZE=50
STEP=.02
a=load_iris()
x=a['data']
y=a['target']
train,test,train_lab,test_lab=train_test_split(x,y,test_size=.20,random_state=22)
def plot(train,train_lab):
    for index,row in enumerate(train):
        if train_lab[index]==0:
            c='r'
	    marker='>'
    	    q=plt.scatter(train[index,0],train[index,1],c=c,marker=marker,s=SIZE)
        elif train_lab[index]==1:
            c='b'
	    marker='s'
    	    w=plt.scatter(train[index,0],train[index,1],c=c,marker=marker,s=SIZE)
        else:
            c='g'
	    marker='+'
    	    e=plt.scatter(train[index,0],train[index,1],c=c,marker=marker,s=SIZE)
    plt.legend((q,w,e),('setosa','virginica','versicolor'),scatterpoints=1)

#Uncomment to see the plot
#plot(train,train_lab)
#plt.show()
clf=DecisionTreeClassifier()
clf.fit(train,train_lab)
output=clf.predict(test)
print accuracy_score(test_lab,output)
pca=PCA(n_components=2,whiten=True)
new_train=pca.fit_transform(train)
mean=new_train.mean(axis=0)
std=new_train.std(axis=0)
new_train=(new_train-mean)/std
clf.fit(new_train,train_lab)
#Uncomment to see the plot
#plot(new_train,train_lab)
x_min,x_max=new_train[:,0].min()-1,new_train[:,0].max()+1
y_min,y_max=new_train[:,1].min()-1,new_train[:,1].max()+1

xx,yy=np.meshgrid(np.arange(x_min,x_max,STEP),np.arange(y_min,y_max,STEP))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
cs=plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
plot(new_train,train_lab)
plt.show()
