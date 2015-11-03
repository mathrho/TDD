
"""
Script to train a basic action classification system.

Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
This script is used to experimentally test different parameter settings for the SVMs.

"""

import os, sys, collections, random, string
import numpy as np
import scipy.io
import argparse
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
import classify_library
import cPickle as pickle


# Path to the video repository
UCF101_DIR = '/home/zhenyang/Workspace/data/UCF101'
train_list = '/home/zhenyang/Workspace/data/UCF101/train1.txt'
test_list = '/home/zhenyang/Workspace/data/UCF101/test1.txt'

def make_FV_matrix(videos, features, labels):

    matrix = []
    target = []

    for i,video in enumerate(videos):
        print video
        vid_name = os.path.splitext(video)[0]

        fvs = []
        for feature in features:
            vid_file = os.path.join(feature,vid_name)
            if not os.path.exists(vid_file+'.fv.mat'):
                print '%s: Feature does not exist in %s, skip this video!' % (video,feature)
                break
            matfile = scipy.io.loadmat(vid_file+'.fv.mat')
            fv = matfile['fv']

            if fv.size:
                # power-normalization
                fv = np.sign(fv) * (np.abs(fv) ** 0.5)
                # L2 normalize
                norms = np.sqrt(np.sum(fv ** 2))
                fv /= norms
                fv[np.isnan(fv)] = 100

                fvs.append(fv)

        if len(fvs) == len(features):
            # concatenate fvs
            output_fv = np.hstack(fvs)

            # L2 normalize the entire fv.
            norm = np.sqrt(np.sum(output_fv ** 2))
            output_fv /= norm

            matrix.append(output_fv)
            target.append(labels[i])

    X = np.vstack(matrix)
    Y = np.array(target)

    return (X,Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FisherExtractior')
    parser.add_argument('-f', '--features', dest='features', help='Specify features to use for classification.', type=str)

    args = parser.parse_args()

    #FV_DIR = '/home/zhenyang/Workspace/data/UCF101/features/fv'
    features = args.features.split(',')
    all_features = '_'.join(features)
    FV_DIRs = [os.path.join(UCF101_DIR, feature) for feature in features]

    f = open(train_list, 'r')
    videos = f.readlines()
    f.close()
    videos_train = [line.split()[0] for line in [video.rstrip() for video in videos]]
    labels_train = [int(line.split()[1]) for line in [video.rstrip() for video in videos]]

    f = open(test_list, 'r')
    videos = f.readlines()
    f.close()
    videos_test = [line.split()[0] for line in [video.rstrip() for video in videos]]
    labels_test = [int(line.split()[1]) for line in [video.rstrip() for video in videos]]

    # GET THE TRAINING AND TESTING DATA.
    flname_train = '/home/zhenyang/Workspace/data/UCF101/features/UCF101_train1_%s.fv' % (all_features,)
    if os.path.exists(flname_train+'.npz'):
        data = np.load(flname_train+'.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
    else:
        X_train, Y_train = make_FV_matrix(videos_train, FV_DIRs, labels_train)
        np.savez(flname_train, X_train=X_train, Y_train=Y_train)

    flname_test = '/home/zhenyang/Workspace/data/UCF101/features/UCF101_test1_%s.fv' % (all_features,)
    if os.path.exists(flname_test+'.npz'):
        data = np.load(flname_test+'.npz')
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        X_test, Y_test = make_FV_matrix(videos_test, FV_DIRs, labels_test)
        np.savez(flname_test, X_test=X_test, Y_test=Y_test)

    # TRAINING
    model_file = '/home/zhenyang/Workspace/data/UCF101/models/UCF101_linearsvm_train1_%s.model' % (all_features,)
    if os.path.exists(model_file+'.pkl'):
        with open(model_file+'.pkl', 'r') as fp:
            classifier = pickle.load(fp)

    else:
        estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
        classifier = estimator.fit(X_train, Y_train)
        # store the model in a pickle file
        with open(model_file+'.pkl', 'w') as fp:
            pickle.dump(classifier, fp)

    # TESTING
    result_file = '/home/zhenyang/Workspace/data/UCF101/results/UCF101_linearsvm_train1_test1_%s.result' % (all_features,)
    metrics = classify_library.metric_scores(classifier, X_test, Y_test, verbose=True)
    print metrics

