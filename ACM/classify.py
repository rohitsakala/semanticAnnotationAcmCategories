'''
    Analyzing Various Classification Algorithms on the Created Model
'''

# loading gensim modules
from gensim.models import Doc2Vec

# loading numpy
import numpy

# loading random
#from random import shuffle

# loading classifiers
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#For String Conversion of Labels
from sklearn import preprocessing

# Evaluation Metrics
from sklearn.metrics import f1_score

# Random package for testing
from random import randint

import pickle

def classification_report(classifier, feature_array, labels, test_arrays, \
    test_labels, algorithm, model):
    '''
        Create Classification Report
    '''
    classifier.fit(feature_array, labels)

    # Computing accuracy
    accuracy_score = classifier.score(test_arrays, test_labels)

    # Computing F1-score
    clf_f1_score = f1_score(test_labels, classifier.predict(test_arrays), average="micro")

    print("--------------", algorithm, ": ", model, "--------------")
    print("Accuracy Score: ".ljust(18), accuracy_score)
    print("F1-score: ".ljust(18), clf_f1_score)


def classify(args):

    '''
        Apply Various Classification Algorithms
    '''

    train_arrays_cow, train_labels_cow, \
    test_arrays_cow, test_labels_cow, \
    train_arrays_skip, train_labels_skip, \
    test_arrays_skip, test_labels_skip = args

    classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, \
        fit_intercept=True, intercept_scaling=1, penalty='l2', \
        random_state=None, tol=0.0001)

    # Logistic Regression Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, "Logistic Regression", "DM Model")

    # Logistic Regression Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip, "Logistic Regression", "DBOM Model")


    classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    # SVM Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, "SVM Classifier", "DM Model")

    # SVM Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip, "SVM Classifier", "DBOM Model")


    classifier = KNeighborsClassifier(n_neighbors=5)

    # KNN Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, "KNN Classifier", "DM Model")

    # KNN Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip, "KNN Classifier", "DBOM Model")

    classifier = RandomForestClassifier(n_estimators=10, criterion='gini', \
        max_depth=None, min_samples_split=2, min_samples_leaf=1, \
        min_weight_fraction_leaf=0.0, max_features='auto', \
        max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, \
        random_state=None, verbose=0, warm_start=False, class_weight=None)

    # Random Forest Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, \
        "Random Forest Classifier", "DM Model")

    # Random Forest Classifier Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip, \
        "Random Forest Classifier", "DBOM Model")

def load_model():
    '''
        Loading and Building Train and Test Data
    '''
    #loading labels
    labels = pickle.load(open('labels.p', 'rb'))


    #Using LabelEncoder to convert string to numerical value.
    label_encoder = preprocessing.LabelEncoder()
    vec_label_train = label_encoder.fit_transform( labels[:549] )
    vec_label_test = label_encoder.transform( labels[549:] ) 

    # learning model Distributed memory model
    model = Doc2Vec.load('./acm_cow.d2v')

    # initialising training vectors
    train_arrays_cow = numpy.zeros((549, 100))
    train_labels_cow = numpy.zeros(549)

    #print(len(model.docvecs))

    # updating training arrays
    for i in range(549):
        prefix_train_pos = "TRAIN_" + str(i)
        train_arrays_cow[i] = model.docvecs[prefix_train_pos]
        train_labels_cow[i] = vec_label_train[i]
        
    # initialising testing vectors
    test_arrays_cow = numpy.zeros((549, 100))
    test_labels_cow = numpy.zeros(549)

    # updating testing arrays
    for i in range(549):
        prefix_test_pos = 'TEST_' + str(i)
        test_arrays_cow[i] = model.docvecs[prefix_test_pos]
        test_labels_cow[i] = vec_label_test[i]

    # learning model Distributed Bag of words model
    model = Doc2Vec.load('./acm_skip.d2v')

    # initialising training vectors
    train_arrays_skip = numpy.zeros((549, 100))
    train_labels_skip = numpy.zeros(549)

    # updating training arrays
    for i in range(549):
        prefix_train_pos = 'TRAIN_' + str(i)
        train_arrays_skip[i] = model.docvecs[prefix_train_pos]
        train_labels_skip[i] = vec_label_train[i]
        
    # initialising testing vectors
    test_arrays_skip = numpy.zeros((549, 100))
    test_labels_skip = numpy.zeros(549)

    # updating testing arrays
    for i in range(549):
        prefix_test_pos = 'TEST_' + str(i)
        test_arrays_skip[i] = model.docvecs[prefix_test_pos]
        test_labels_skip[i] = vec_label_test[i]
        
    to_return = (train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, \
        train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip)

    return to_return

def main():
    '''
        Load and Classify Model.
    '''
    classify(load_model())

if __name__ == '__main__':
    main()
