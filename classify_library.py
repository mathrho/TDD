
# coding: utf-8

## Trains a One vs. Rest SVM classifier on the fisher vector video outputs.

import os, sys, collections, random, string
import numpy as np
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


################################################################
# Useful Helper functions                                      #
################################################################


#Returns the Mean Average Precision (mAP) to evaluate the performance of a run
#Arguments:
# 1. classifier such as sklearn.multiclass.OneVsRestClassifier
# 2. X_test: data to classify
# 3. Y_test: class labels.
# Returns: (mAP, [aps])
def metric_mAP(classifier, X_test, Y_test, verbose=False):
    estimators = classifier.estimators_
    classes = classifier.classes_
    aps = []
    for estimator,class_num in zip(estimators, classes):
        aps.append(metric_AP(estimator, class_num, X_test, Y_test, verbose=verbose))
    map_val = np.mean(aps)
    if verbose: print "mean AP = %.3f" % map_val
    return map_val


#Average Precision
def metric_AP(estimator, class_num, X_test, Y_test, verbose=False):
    
    scores = estimator.decision_function(X_test)
    #Sorted list of (original_index,score) tuples.
    scores_sorted = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)
    # collect the positive results in the dataset
    positive_ranks = [i for i,score in enumerate(scores_sorted) if Y_test[score[0]]==class_num]
    # accumulate trapezoids with this basis
    recall_step = 1.0 / len(positive_ranks)
    ap = 0
    for ntp,rank in enumerate(positive_ranks):
       # ntp = nb of true positives so far
       # rank = nb of retrieved items so far
       # y-size on left side of trapezoid:
       precision_0 = ntp/float(rank) if rank > 0 else 1.0
       # y-size on right side of trapezoid:
       precision_1 = (ntp + 1) / float(rank + 1)
       ap += (precision_1 + precision_0) * recall_step / 2.0
    if verbose: print "class %d, AP = %.3f" % (class_num, ap)
    return ap


#For a sklearn.multiclass.OneVsRestClassifier, calculate the mAP (mean interpolated average precision),
# accuracy score, and average Precision
def metric_scores(classifier, X_test, Y_test, verbose=False):
    mAP = metric_mAP(classifier, X_test, Y_test,verbose=verbose)
    X_test_predictions = classifier.predict(X_test)
    accuracy_score = metrics.accuracy_score(Y_test, X_test_predictions)
    avg_Precision = metrics.precision_score(Y_test, X_test_predictions, average='macro')
    avg_Recall = metrics.recall_score(Y_test, X_test_predictions, average='macro')
    return (mAP, accuracy_score, avg_Precision, avg_Recall)


#Helper methods for plotting metrics.
#plot_confusion_matrix(Y_test, X_test_predictions)
def plot_confusion_matrix(y_test, y_pred):# Compute confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)

    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

