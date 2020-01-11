''' This is an example of Coursera course in Machine Learning
using Python.

This is using scikit=learn library '''

''' The next two lines are use when data set have different scales
or outliers and fix them. Transform raw feature vectors into a suitable
form of vector for modeling. '''

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

'''To split dataset into train and test sets. Scikit separes this data
randomly'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

'''Setup algorithm. E.g. build a classifier using a support vector
classification algorithm, we call our estimator instance CLF and initialize
its parameters. '''

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

'''Training using the train set to the fit method. The CLF model learns
to classify unknown cases.'''

clf.fit(X_train, y_train)

'''Test set to run predictions and the result tells us what the class of each
unknown value is. '''

clf.predict(X_test)

'''Use different metrics to evaluate your model accuracy. E.g. using a
confusion matrix to show the results.'''

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, yhat, labels=[1,0]))

''' Save the model'''

import pickle
s = pickle.dumps(clf)
