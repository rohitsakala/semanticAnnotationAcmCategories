################ This code is written to analyse various classification algorithms #######################

# loading gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# loading numpy
import numpy

# loading random
from random import shuffle

# loading classifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# learning model Distributed memory model
model_cow = Doc2Vec.load('./imdb_cow.d2v')

# initialising training vectors
train_arrays_cow = numpy.zeros((25000, 100))
train_labels_cow = numpy.zeros(25000)

# updating training arrays
for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays_cow[i] = model_cow.docvecs[prefix_train_pos]
    train_arrays_cow[12500 + i] = model_cow.docvecs[prefix_train_neg]
    train_labels_cow[i] = 1
    train_labels_cow[12500 + i] = 0

# initialising testing vectors
test_arrays_cow = numpy.zeros((25000, 100))
test_labels_cow = numpy.zeros(25000)

# updating testing arrays
for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays_cow[i] = model_cow.docvecs[prefix_test_pos]
    test_arrays_cow[12500 + i] = model_cow.docvecs[prefix_test_neg]
    test_labels_cow[i] = 1
    test_labels_cow[12500 + i] = 0

# learning model Distributed Bag of words model
model_skip = Doc2Vec.load('./imdb_skip.d2v')

# initialising training vectors
train_arrays_skip = numpy.zeros((25000, 100))
train_labels_skip = numpy.zeros(25000)

# updating training arrays
for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays_skip[i] = model_skip.docvecs[prefix_train_pos]
    train_arrays_skip[12500 + i] = model_skip.docvecs[prefix_train_neg]
    train_labels_skip[i] = 1
    train_labels_skip[12500 + i] = 0

# initialising testing vectors
test_arrays_skip = numpy.zeros((25000, 100))
test_labels_skip = numpy.zeros(25000)

# updating testing arrays
for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays_skip[i] = model_skip.docvecs[prefix_test_pos]
    test_arrays_skip[12500 + i] = model_skip.docvecs[prefix_test_neg]
    test_labels_skip[i] = 1
    test_labels_skip[12500 + i] = 0

# Training logistic regression classifier for DM model
classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
classifier.fit(train_arrays_cow, train_labels_cow)

# Computing accuracy
model_cow_logistic = classifier.score(test_arrays_cow, test_labels_cow)

# Training logistic regression classifier for DBOM
classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
classifier.fit(train_arrays_skip, train_labels_skip)

# Computing accuracy
model_skip_logistic = classifier.score(test_arrays_skip, test_labels_skip)

# Training svm classifier for DM model
classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
classifier.fit(train_arrays_cow, train_labels_cow)

# Computing accuracy
model_cow_svm = classifier.score(test_arrays_cow, test_labels_cow)

# Training svm classifier for DBOM
classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
classifier.fit(train_arrays_skip, train_labels_skip)

# Computing accuracy
model_skip_svm = classifier.score(test_arrays_skip, test_labels_skip)

# Training knn classifier for DM model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_arrays_cow, train_labels_cow)

# Computing accuracy
model_cow_knn = classifier.score(test_arrays_cow, test_labels_cow)

# Training knn classifier for DBOM
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_arrays_skip, train_labels_skip)

# Computing accuracy
model_skip_knn = classifier.score(test_arrays_skip, test_labels_skip)

# Training random forest classifier for DM model
classifier = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
classifier.fit(train_arrays_cow, train_labels_cow)

# Computing accuracy
model_cow_rf = classifier.score(test_arrays_cow, test_labels_cow)

# Training random forest classifier for DBOM
classifier = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
classifier.fit(train_arrays_skip, train_labels_skip)

# Computing accuracy
model_skip_rf = classifier.score(test_arrays_skip, test_labels_skip)

# Print all accuracies
print("Algorithm","Model","Accuracy")
print("Logistic Regression","Skip",model_skip_logistic)
print("Logistic Regression","Cow",model_cow_logistic)
print("RBF Kernel Svm","Skip",model_skip_svm)
print("RBF Kernel Svm","Cow",model_cow_svm)
print("KNN","Skip",model_skip_knn)
print("KNN","Cow",model_cow_knn)
print("KNN","Skip",model_skip_rf)
print("KNN","Cow",model_cow_rf)