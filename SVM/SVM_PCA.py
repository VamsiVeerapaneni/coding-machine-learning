
# Accuracy in Kaggle 0.98243

# Import the necessary Packages

from utility import *
Cross_validation=True

# Choose the number of Principle Axis
axis=35
print "Number of axis is set"

train,train_label,test=load_data()
print "Train and test set loaded"
new_train,new_test=perform_PCA(train,test,axis)
print "PCA on train and test set is complete"

if Cross_validation:
	print "Splitting into cross validation set"
	train,test,train_label,test_label=train_test_split(new_train,train_label,test_size=.20,random_state=32)
	clf=SVC()
	print "Fitting the SVM"
	clf.fit(train,train_label)
	print "Predicting the output for the test instances"
	output=clf.predict(test)
	print "Accuracy on the validation set is : ",accuracy_score(test_label,output)
else:
	# Build a SVM Classifier
	clf=SVC()
	clf.fit(new_train,train_label)
	output=clf.predict(test)

	# Store the output

	s=pd.read_csv('../data/final.csv')
	s.Label=output
	s.to_csv('../data/final.csv',index=False)
