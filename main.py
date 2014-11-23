import sys
import math

import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

import csv
import numpy as np
import pylab as pl
from sklearn import svm, datasets, cross_validation
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

training = 'T-61_3050_training_dataset.csv'
test = 'T-61_3050_test_dataset.csv'

def read_csv(csv_file, has_header=True):
    targetColorArray = []
    targetQualityArray = []
    dataArray = []
    header = []
    with open(csv_file, 'r') as File:
        Reader = csv.reader(File, delimiter = ',')
        # Get the header first if any.
        if has_header:
            header = Reader.next()
        for row in Reader:
            targetColorArray.append(np.array(row[-1]))
            targetQualityArray.append(np.array(row[-2], dtype=int))
            dataArray.append(np.array(row[:-2], dtype='f'))
    targetColorArray = np.array(targetColorArray)
    targetQualityArray = np.array(targetQualityArray)
    dataArray = np.vstack(dataArray)
    if has_header:
        return (header, dataArray, targetQualityArray, targetColorArray)
    return (dataArray, targetQualityArray, targetColorArray)

(header, trainData, trainQuality, trainColor) = read_csv(training)
(_, testData, testQuality, testColor) = read_csv(test)

def count_equal(y1, y2):
    matched = 0
    for i,t in enumerate(y2):
        if y1[i] == t:
            matched += 1
    return matched

def print_results(clf, trainData, trainTarget, testData, testTarget):
    print 'Classifier', clf
    testPredict = clf.predict(testData)
    test_match = count_equal(testPredict, testTarget)
    print 'Matched in test set', test_match / float(len(testTarget))
    print 'F1 score in test set', f1_score(testTarget, testPredict,
    pos_label=None)
    trainPredict = clf.predict(trainData)
    train_match = count_equal(trainPredict, trainTarget)
    print 'Matched in training set', train_match / float(len(trainTarget))
    print 'F1 score in training set', f1_score(trainTarget, trainPredict,
    pos_label=None)

def plot_features(x, y, labels, axis_x, axis_y):
    pl.figure(1)
    pl.clf()
    pl.xlabel(axis_x)
    pl.ylabel(axis_y)
    for label in np.unique(labels):
        indexes = np.where(labels == label)[0]
        pl.plot(x[indexes], y[indexes], '.', label=label)
    pl.legend()
    pl.show()

def plot_features3d(x, y, z, labels, axis_x, axis_y, axis_z):
    fig = pl.figure(1)
    pl.clf()
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(labels):
        indexes = np.where(labels == label)[0]
        ax.plot(x[indexes], y[indexes], z[indexes], '.', label=label)
    pl.legend()
    pl.show()

def plot_data():
    # Try finding correlations about color/quality for pair of features.
    # Observe if any separation between them.
    #for i in range(trainData.shape[1]):
    #    for j in range(i+1, trainData.shape[1]):
    #        plot_features(trainData[:,i], trainData[:,j], trainColor, header[i], header[j])
    for i in range(trainData.shape[1]):
        for j in range(i+1, trainData.shape[1]):
            plot_features(trainData[:,i], trainData[:,j], trainQuality, header[i], header[j])

def scale(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

def train_predict(clf_constructor):
    print '== Color accuracy =='
    clf = clf_constructor()
    clf.fit(trainData, trainColor)
    print_results(clf, trainData, trainColor, testData, testColor)
    print '== Quality accuracy =='
    clf = clf_constructor()
    clf.fit(trainData, trainQuality)
    print_results(clf, trainData, trainQuality, testData, testQuality)

def train_predict_two_clf_one_each_type(clf_constructor):
    print '== Color accuracy =='
    # Train classifier for color first.
    # This has a pretty good accuracy, > 90%.
    clf_type = clf_constructor()
    clf_type.fit(trainData, trainColor)
    print_results(clf_type, trainData, trainColor, testData, testColor)
    print '== Quality accuracy =='
    # Init two classifiers, one for each type of the wine (red/white).
    clfs = {label: clf_constructor() for label in np.unique(trainColor)}
    # Train them with separate data, depending on the predicted type of wine.
    for label in np.unique(trainColor):
        indexes = np.where(trainColor == label)[0]
        clfs[label].fit(trainData[indexes], trainQuality[indexes])
    # Predict the color type of wine for test data and then use the
    # classifier trained for that color type as final answer.
    testPredictType = clf_type.predict(testData)
    # Initialize empty array which we'll fill as we find them out.
    # Represents the prediction of quality for entire test set.
    testPredictQuality = np.empty(len(testQuality))
    for label in np.unique(testPredictType):
        indexes = np.where(testPredictType == label)[0]
        y_pred = clfs[label].predict(testData[indexes])
        testPredictQuality[indexes] = y_pred
    test_match = count_equal(testPredictQuality, testQuality)
    print 'Matched in test set', test_match / float(len(testQuality))
    print 'F1 score in test set', f1_score(testQuality, testPredictQuality, pos_label=None)

def main(args):
    #global trainData, testData
    #trainData = scale(trainData)
    #testData = scale(testData)

    if args[0] == 'plot':
        plot_data()

    # See if features have gaussian-like distribution and maybe adjust them.
    if args[0] == 'plot_normal_distribution':
        # x0 <- log(x0), x1 <- log(x1)
        pl.hist(trainData[:,1], bins=100)
        pl.show()

    if args[0] == 'naive_bayes':
        train_predict(lambda: GaussianNB())
        train_predict_two_clf_one_each_type(lambda: GaussianNB())
    if args[0] == 'knn':
        train_predict(lambda: KNeighborsClassifier(n_neighbors=5))
        train_predict_two_clf_one_each_type(lambda: KNeighborsClassifier(n_neighbors=5))
    if args[0] == 'svm':
        train_predict(lambda: svm.SVC(gamma=0.001, C=100.))
        train_predict_two_clf_one_each_type(lambda: svm.SVC(gamma=0.001, C=100.))

    if args[0] == 'pca3d':
        pca = PCA(n_components=3)
        X = pca.fit_transform(trainData)
        plot_features3d(X[:,0], X[:,1], X[:,2], trainColor, 'pca0', 'pca1',
        'pca2')
    if args[0] == 'pca2d':
        pca = PCA(n_components=2)
        X = pca.fit_transform(trainData)
        plot_features(X[:,0], X[:,1], trainQuality, 'pca0', 'pca1')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Please specify a classifier type'
        sys.exit(-1)
    main(sys.argv[1:])
