
# Only SVM with accuracy 74% in Kaggle
from utility import *

Cross_validation=True
train,train_label,test=load_data()

print "Train and the Test set is Loaded!!"

# Build a SVM Classifier
clf=SVC()
print "Building SVM using training set and its labels"

if Cross_validation:
	train,test,train_label,test_label=train_test_split(train,train_label,test_size=.20,random_state=32)
	clf.fit(train,train_label)
	print "Predicting the output for the test instances"
	output=clf.predict(test)
	print "Accuracy on the validation set is : ",accuracy_score(test_label,output)
else:
	clf.fit(train,train_label)
	print "Predicting the output for the test instances"
	output=clf.predict(test)
	# Store the output
	print "Writing back to the output file"
	s=pd.read_csv('../data/final.csv')
	s.Label=output
	s.to_csv('../data/final.csv',index=False)
