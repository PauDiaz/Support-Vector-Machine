#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
# from sklearn import svm
# from sklearn.metrics import accuracy_score

# def email_identifier_SVM():
	
# 	clf=svm.SVC(kernel='linear') # create my classifier


# 	t0 = time() 
# 	clf.fit(features_train,labels_train) #fit the training features with the training labels
# 	print "Training time:", round(time()-t0, 3), "s" # Time to train the classifier

# 	t1 = time() 
# 	pred=clf.predict(features_test)
# 	print "Predicting time:", round(time()-t1, 3), "s" # Time to make predictions

# 	# Set the accuracy of the algorithm and print it
# 	accuracy= round(accuracy_score(labels_test, pred),4)
# 	print "The accuracy score is:",accuracy

# email_identifier_SVM()

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 165.905 s
# Predicting time: 18.949 s
# The accuracy score is: 0.9841


#########################################################

# from sklearn import svm
# from sklearn.metrics import accuracy_score

# def email_identifier_SVM_tradeoff():
# 	'''One way to speed up an algorithm is to train it on a smaller training dataset. 
# 	The tradeoff is that the accuracy almost always goes down when
# 	 you do this.'''

# 	clf=svm.SVC(kernel='linear') # create my classifier
	
	
# 	features_train_sliced = features_train[:len(features_train)/100] 
# 	labels_train_sliced = labels_train[:len(labels_train)/100]

# 	#These lines slice the training dataset down 
# 	#to 1% of its original size, tossing out 99% of the training 
# 	#data. You can leave all other code unchanged.
	
# 	t0 = time() 
# 	clf.fit(features_train_sliced,labels_train_sliced) #fit the training features with the training labels
# 	print "Training time:", round(time()-t0, 3), "s" # Time to train the classifier

# 	t1 = time() 
# 	pred=clf.predict(features_test)
# 	print "Predicting time:", round(time()-t1, 3), "s" # Time to make predictions

# 	# Set the accuracy of the algorithm and print it
# 	accuracy= round(accuracy_score(labels_test, pred),4)
# 	print "The accuracy score is:",accuracy



# email_identifier_SVM_tradeoff()

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 0.111 s
# Predicting time: 1.201 s
# The accuracy score is: 0.8845


##########################     Analysis     #############################

# By slicing the features and the labels to train to a 1% of the 
# original size we got a reduction of the accuracy from  0.9841 
# to 0.8845 (around 10%). On the other hand, the trining time also 
# decreased from 165.905 s to 0.111 (around 99.93%) and the predicting
# time followed the same pattern from 18.949 s to 1.201 s (93.66%).


##########################    end     ###################################


#########################################################################


# KERNEL = rbf



# from sklearn import svm
# from sklearn.metrics import accuracy_score

# def email_identifier_SVM_tradeoff_rbf():
# 	''' Keep the training set slice code from the last quiz, so that you are still 
# 	training on only 1percent of the full training set. Change the kernel of your SVM 
# 	rbf. What is the accuracy now, with this more complex kernel?'''

# 	clf=svm.SVC(kernel='rbf') # create my classifier
	
	
# 	features_train_sliced = features_train[:len(features_train)/100] 
# 	labels_train_sliced = labels_train[:len(labels_train)/100]

# 	#These lines slice the training dataset down 
# 	#to 1% of its original size, tossing out 99% of the training 
# 	#data. You can leave all other code unchanged.
	
# 	t0 = time() 
# 	clf.fit(features_train_sliced,labels_train_sliced) #fit the training features with the training labels
# 	print "Training time:", round(time()-t0, 3), "s" # Time to train the classifier

# 	t1 = time() 
# 	pred=clf.predict(features_test)
# 	print "Predicting time:", round(time()-t1, 3), "s" # Time to make predictions

# 	# Set the accuracy of the algorithm and print it
# 	accuracy= round(accuracy_score(labels_test, pred),4)
# 	print "The accuracy score is:",accuracy



# email_identifier_SVM_tradeoff_rbf()

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 0.116 s
# Predicting time: 1.236 s
# The accuracy score is: 0.616


#########################################################################

# KERNEL = rbf
# C = 10.0, C = 100.0, C = 1000.0, C = 10000.0


# from sklearn import svm
# from sklearn.metrics import accuracy_score

# def email_identifier_SVM_tradeoff_rbf_C():
# 	''' Keep the training set slice code from the last quiz, so that you are still 
# 	training on only 1percent of the full training set. Change the kernel of your SVM 
# 	rbf. What is the accuracy now, with this more complex kernel?'''

# 	clf=svm.SVC(C=10000.0,kernel='rbf') # create my classifier
	
	
# 	features_train_sliced = features_train[:len(features_train)/100] 
# 	labels_train_sliced = labels_train[:len(labels_train)/100]
	
# 	t0 = time() 
# 	clf.fit(features_train_sliced,labels_train_sliced) #fit the training features with the training labels
# 	print "Training time:", round(time()-t0, 3), "s" # Time to train the classifier

# 	t1 = time() 
# 	pred=clf.predict(features_test)
# 	print "Predicting time:", round(time()-t1, 3), "s" # Time to make predictions

# 	# Set the accuracy of the algorithm and print it
# 	accuracy= round(accuracy_score(labels_test, pred),4)
# 	print "The accuracy score is:",accuracy



# email_identifier_SVM_tradeoff_rbf_C()



#######        C=10.0     ################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 0.117 s
# Predicting time: 1.231 s
# The accuracy score is: 0.616
##########################################

#######        C=100.0    ################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 0.119 s
# Predicting time: 1.244 s
# The accuracy score is: 0.616
##########################################

#######        C=1000.0    ###############
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 0.123 s
# Predicting time: 1.202 s
# The accuracy score is: 0.8214
##########################################

#######        C=10000.0    ##############
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 0.113 s
# Predicting time: 1.005 s
# The accuracy score is: 0.8925
##########################################


#########################################################################

# Now that you have optimized C for the RBF kernel, go back to using the full 
# training set. In general, having a larger training set will improve the 
# performance of your algorithm, so (by tuning C and training on a large
# dataset) we should get a fairly optimized result. What is the accuracy 
# of the optimized SVM


# from sklearn import svm
# from sklearn.metrics import accuracy_score

# def email_identifier_SVM_rbf_C_10000():
# 	''' Keep the training set slice code from the last quiz, so that you are still 
# 	training on only 1percent of the full training set. Change the kernel of your SVM 
# 	rbf. What is the accuracy now, with this more complex kernel?'''

# 	clf=svm.SVC(C=10000.0,kernel='rbf') # create my classifier
	
	
# 	#features_train_sliced = features_train[:len(features_train)/100] 
# 	#labels_train_sliced = labels_train[:len(labels_train)/100]
	
# 	t0 = time() 
# 	clf.fit(features_train,labels_train) #fit the training features with the training labels
# 	print "Training time:", round(time()-t0, 3), "s" # Time to train the classifier

# 	t1 = time() 
# 	pred=clf.predict(features_test)
# 	print "Predicting time:", round(time()-t1, 3), "s" # Time to make predictions

# 	# Set the accuracy of the algorithm and print it
# 	accuracy= round(accuracy_score(labels_test, pred),4)
# 	print "The accuracy score is:",accuracy



# email_identifier_SVM_rbf_C_10000()

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 113.898 s
# Predicting time: 11.157 s
# The accuracy score is: 0.9909
#########################################################################


# What class does your SVM (0 or 1, corresponding to Sara and Chris respectively) 
# predict for element 10 of the test set? The 26th? The 50th? (Use the RBF kernel, 
# 	C=10000, and 1% of the training set. Normally you'd get the best results using 
# 	the full training set, but we found that using 1% sped up the computation considerably 
# 	and did not change our results--so feel free to use that shortcut here.)


# from sklearn import svm
# from sklearn.metrics import accuracy_score

# def email_identifier_SVM_tradeoff_rbf_C_10000():
# 	''' Keep the training set slice code from the last quiz, so that you are still 
# 	training on only 1percent of the full training set. Change the kernel of your SVM 
# 	rbf. What is the accuracy now, with this more complex kernel?'''

# 	clf=svm.SVC(C=10000.0,kernel='rbf') # create my classifier
	
	
# 	features_train_sliced = features_train[:len(features_train)/100] 
# 	labels_train_sliced = labels_train[:len(labels_train)/100]
	
# 	t0 = time() 
# 	clf.fit(features_train_sliced,labels_train_sliced) #fit the training features with the training labels
# 	print "Training time:", round(time()-t0, 3), "s" # Time to train the classifier

# 	t1 = time() 
# 	pred=clf.predict(features_test)
# 	print "Predicting time:", round(time()-t1, 3), "s" # Time to make predictions

# 	# Set the accuracy of the algorithm and print it
# 	accuracy= round(accuracy_score(labels_test, pred),4)
# 	print "The accuracy score is:",accuracy

	

# 	print "the 10th is",pred[10], ". The 26th is: ",pred[26], ". And the 50th is: ",pred[50]

# email_identifier_SVM_tradeoff_rbf_C_10000()


# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 0.108 s
# Predicting time: 1.003 s
# The accuracy score is: 0.8925
# the 10th is 1 . The 26th is:  0 . And the 50th is:  1
#########################################################################


# There are over 1700 test events. How many are predicted to be in the 
# Chris (1) class? (Use the RBF kernel, C=10000., and the full training set.)


from sklearn import svm
from sklearn.metrics import accuracy_score

def email_identifier_SVM_rbf_C_10000():
	''' Keep the training set slice code from the last quiz, so that you are still 
	training on only 1percent of the full training set. Change the kernel of your SVM 
	rbf. What is the accuracy now, with this more complex kernel?'''

	clf=svm.SVC(C=10000.0,kernel='rbf') # create my classifier
	
	
	#features_train_sliced = features_train[:len(features_train)/100] 
	#labels_train_sliced = labels_train[:len(labels_train)/100]
	
	t0 = time() 
	clf.fit(features_train,labels_train) #fit the training features with the training labels
	print "Training time:", round(time()-t0, 3), "s" # Time to train the classifier

	t1 = time() 
	pred=clf.predict(features_test)
	print "Predicting time:", round(time()-t1, 3), "s" # Time to make predictions

	# Set the accuracy of the algorithm and print it
	accuracy= round(accuracy_score(labels_test, pred),4)
	print "The accuracy score is:",accuracy

	Chris=sum(pred)
	


	print "The number of emails from Chris is:", Chris


email_identifier_SVM_rbf_C_10000()


# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training time: 110.217 s
# Predicting time: 11.098 s
# The accuracy score is: 0.9909
# The number of emails from Chris is: 877

