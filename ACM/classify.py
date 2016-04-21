#!/usr/bin/env python -W ignore::DeprecationWarning

'''
    Analyzing Various Classification Algorithms on the Created Model
'''

# loading gensim modules
from gensim.models import Doc2Vec

# loading np
import numpy as np

from sklearn.cross_validation import train_test_split

# loading classifiers
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

# String Conversion of Labels to numerical values
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

# Evaluation Metrics
from sklearn.metrics import average_precision_score

from sklearn.multiclass import OneVsRestClassifier

import pickle

# Importing Neural Network Module
from NNet import NeuralNet


def mean_average_precision(ave_prec_list):
    '''
        Calculate the mean of all the precision values in ave_prec_list.

    Parameters
    ----------
    ave_prec_list : array-like
        Ground truth (precision values list).
    Returns
    ----------
    MAP: mean of the values in ave_prec_list.
    '''
    ave_prec_arr = np.array(ave_prec_list)
    return ave_prec_arr.mean()


def calc_mean_average_precision(y_true_list, y_pred_list):
    '''
        Function to calculate Mean Average Precision value,
        given the testing and predicted labels.
    Parameters
    ----------
    y_true_list : array-like
        Ground truth (Testing Labels).
    y_pred_list : array-like
        Ground truth(Predicted Labels)
    Returns
    ----------
    MAP: Mean Average Precision.
    '''
    ave_prec_list = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        ave_prec = average_precision_score(y_true, y_pred,
                                           average="micro", sample_weight=None)
        ave_prec_list.append(ave_prec)
    mean_average_precision_val = mean_average_precision(ave_prec_list)
    return mean_average_precision_val


'''
def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Defines the DCG growth to be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "preprocessing":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    discounts = np.log2(np.arange(len(y_true)) + 2)

    return np.sum(gains / discounts)
'''

def dcg_score(y_true, y_score, k=5):
    
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_score(y_true, y_score, k=5, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    scores = []

    for true_value, pred_value in zip(y_true, y_score):
        best = dcg_score(true_value, pred_value, k)
        actual = dcg_score(true_value, pred_value, k)
        #if best == 0:
        #    best = 0.000000000001
        score = double(actual) / float(best)
        scores.append(score)

    scores = np.array(scores)
    return np.mean(scores)


def classification_report(classifier, feature_array, labels, test_arrays,
        test_labels, algorithm, model, *args, **kwargs):
    '''
        Create Classification Report
    '''

    classifier.fit(feature_array, labels, *args, **kwargs)

    predicted_values = classifier.predict(test_arrays)

    # Computing Mean Average Precision Score
    clf_mean_avg_precision = calc_mean_average_precision(test_labels,
                                                         predicted_values)

    # Computing NDCG Score
    clf_ndcg_score = ndcg_score(test_labels, predicted_values)

    print("--------------", algorithm, ": ", model, "--------------")
    print("MAP value:", clf_mean_avg_precision)
    print("NDCG Value: ", clf_ndcg_score)


def classify(args):
    '''
        Apply Various Classification Algorithms
    '''

    train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, \
        train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip = args

    print('Starting Classification')

    classifier = OneVsRestClassifier(LogisticRegression(C=1.0, \
        class_weight=None, dual=False, fit_intercept=True, \
        intercept_scaling=1, penalty='l2', \
        random_state=None, tol=0.0001))

    # Logistic Regression Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow,
        test_arrays_cow, test_labels_cow, "Logistic Regression", "DM Model")

    classifier = OneVsRestClassifier(LogisticRegression(C=1.0, \
        class_weight=None, dual=False, fit_intercept=True, \
        intercept_scaling=1, penalty='l2', \
        random_state=None, tol=0.0001))

    # Logistic Regression Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, \
        train_labels_skip, test_arrays_skip, test_labels_skip, \
        "Logistic Regression", "DBOM Model")

    classifier = OneVsRestClassifier(LinearSVC(random_state=0))

    # MultinomialNB Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, "LinearSVC Classifier", "DM Model")

    classifier = OneVsRestClassifier(LinearSVC(random_state=0))

    # MultinomialNB Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip, "LinearSVC Classifier", \
        "DBOM Model")

    # Neural Network
    classifier = NeuralNet(50, learn_rate=1e-2)
    maxiter = 10000
    batch = 1500

    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, "Neural Networks", "DM Model", \
        fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)

    classifier = NeuralNet(50, learn_rate=1e-2)

    classification_report(classifier, train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip, "Neural Networks", "DBOM Model", \
        fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)

    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000, \
        criterion='gini', max_depth=None, min_samples_split=2, \
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', \
        max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, \
        random_state=None, verbose=0, warm_start=False, class_weight=None))

    # Random Forest Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, \
        "Random Forest Classifier", "DM Model")

    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000, \
        criterion='gini', max_depth=None, min_samples_split=2, \
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', \
        max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, \
        random_state=None, verbose=0, warm_start=False, class_weight=None))

    # Random Forest Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip, \
        test_arrays_skip, test_labels_skip, \
        "Random Forest Classifier", "DBOM Model")

    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))

    # KNN Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, "KNN Classifier", "DM Model")

    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))

    # KNN Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip,
        test_arrays_skip, test_labels_skip, "KNN Classifier", "DBOM Model")

    classifier = OneVsRestClassifier(GradientBoostingClassifier(
        n_estimators=15000, learning_rate=0.1, max_depth=100,
        random_state=0, loss='deviance', max_features=None,
        verbose=0, max_leaf_nodes=None, warm_start=False,
        presort='auto'))

    # Gradient Boosting Classifier for DM model
    classification_report(classifier, train_arrays_cow, train_labels_cow, \
        test_arrays_cow, test_labels_cow, \
        "Gradient Boosting Classifier", "DM Model")

    classifier = OneVsRestClassifier(GradientBoostingClassifier(
        n_estimators=15000, learning_rate=0.1, max_depth=100,
        random_state=0, loss='deviance', max_features=None,
        verbose=0, max_leaf_nodes=None, warm_start=False,
        presort='auto'))

    # Gradient Boosting Classifier for DBOM model
    classification_report(classifier, train_arrays_skip, train_labels_skip,
        test_arrays_skip, test_labels_skip, \
        "Gradient Boosting Classifier", "DBOM Model")


def load_model():
    '''
        Loading and Building Train and Test Data
    '''
    # loading labels
    labels = pickle.load(open('labels.p', 'rb'))

    # Using LabelEncoder to convert string to numerical value.
    label_encoder = preprocessing.LabelEncoder()
    transformed_labels = label_encoder.fit_transform(labels)

    transformed_labels = np.array(transformed_labels)

    transformed_labels = label_binarize(transformed_labels,
                                        np.unique(transformed_labels))

    print('Found %d Labels' % len(label_encoder.classes_))
    print('Labels:', label_encoder.classes_)

    # initialising feature array
    cow_arrays = np.zeros((247543, 300))

    # learning model Distributed memory model
    model = Doc2Vec.load('./acm_cow.d2v')

    # updating training arrays
    for i in range(247543):
        prefix_train_pos = "SET_" + str(i)
        cow_arrays[i] = model.docvecs[prefix_train_pos]

    train_arrays_cow, test_arrays_cow, train_labels_cow, test_labels_cow = \
        train_test_split(cow_arrays, transformed_labels,
                         test_size=0.1, random_state=42)

    # initialising feature array
    skip_arrays = np.zeros((247543, 300))

    # learning model Distributed Bag of words model
    model = Doc2Vec.load('./acm_skip.d2v')

    # updating training arrays
    for i in range(247543):
        prefix_train_pos = "SET_" + str(i)
        skip_arrays[i] = model.docvecs[prefix_train_pos]

    train_arrays_skip, test_arrays_skip, train_labels_skip, test_labels_skip = \
        train_test_split(skip_arrays, transformed_labels,
                         test_size=0.1, random_state=42)

    to_return = (train_arrays_cow, train_labels_cow,
                 test_arrays_cow, test_labels_cow,
                 train_arrays_skip, train_labels_skip,
                 test_arrays_skip, test_labels_skip)

    return to_return


def main():
    '''
        Load and Classify Model.
    '''
    classify(load_model())

if __name__ == '__main__':
    main()
