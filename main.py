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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

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

def print_results(clf, trainPredict, trainTarget, testPredict, testTarget):
    print 'Classifier', clf
    test_match = count_equal(testPredict, testTarget)
    print 'Matched in test set', test_match / float(len(testTarget))
    print 'F1 score in test set', f1_score(testTarget, testPredict,
    pos_label=None)
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
    return (X - np.mean(X)) / (np.max(X) - np.min(X))

def train_predict(clf_constructor):
    print '== Color accuracy =='
    clf = clf_constructor()
    clf.fit(trainData, trainColor)
    print_results(clf, clf.predict(trainData), trainColor,
                  clf.predict(testData), testColor)
    print '== Quality accuracy =='
    clf = clf_constructor()
    clf.fit(trainData, trainQuality)
    print_results(clf, clf.predict(trainData), trainQuality,
                  clf.predict(testData), testQuality)

def train_predict_two_clf_one_each_type(clf_constructor):
    print '== Color accuracy =='
    # Train classifier for color first.
    # This has a pretty good accuracy, > 90%.
    clf_type = clf_constructor()
    clf_type.fit(trainData, trainColor)
    print_results(clf_type, clf_type.predict(trainData), trainColor,
                  clf_type.predict(testData), testColor)
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
    trainPredictQuality = np.empty(len(trainQuality))
    # Predict training accuracy too.
    for label in np.unique(trainColor):
        indexes = np.where(trainColor == label)[0]
        y_pred = clfs[label].predict(trainData[indexes])
        trainPredictQuality[indexes] = y_pred
    print_results(clfs, trainPredictQuality, trainQuality,
                  testPredictQuality, testQuality)

def make_features_gaussian(x):
    y = np.empty_like(x)
    # I've tried these out on the training set and this is what i came up
    # with.
    functions = {
        0: lambda x: np.log(x),
        1: lambda x: np.power(x, 0.05),
        2: lambda x: np.power(x, 0.65),
        3: lambda x: np.power(x, 0.6),
        4: lambda x: np.power(x, 0.05),
        5: lambda x: np.power(x, 0.4),
        6: lambda x: np.power(x, 0.6),
        7: lambda x: np.power(x, 0.5),
        8: lambda x: x,
        9: lambda x: np.power(x, 0.05),
        10: lambda x: np.power(x, .05),
    }
    for i in range(x.shape[1]):
        y[:,i] = functions[i](x[:,i])
    return y

def pick_k_from_CV(choices_for_k, clf_constructor, X, y, test_size=0.4):
    best_k, best_score = None, None
    # Do it several times as the training split is random.
    i = 0
    while i < 10:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
        test_size=test_size)
        for k in choices_for_k:
            clf = clf_constructor(k)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            if not best_k or best_score < score:
                best_k = k
                best_score = score
        i += 1
    return best_k, best_score

def main(args):
    global trainData, testData
    #trainData = scale(trainData)
    #testData = scale(testData)

    if args[0] == 'plot':
        plot_data()

    # See if features have gaussian-like distribution and maybe adjust them.
    if args[0] == 'plot_normal_distribution':
        # x0 <- log(x0), x1 <- log(x1)
        x = trainData[:,4]
        pl.hist(np.power(x, 0.05), bins=50)
        pl.xlabel('f(x) = x^0.05; f(%s)' % header[4])
        pl.show()

    if args[0] == 'naive_bayes':
        trainData = make_features_gaussian(trainData)
        testData = make_features_gaussian(testData)
        train_predict(lambda: GaussianNB())
        print ''
        train_predict_two_clf_one_each_type(lambda: GaussianNB())
    if args[0] == 'knn':
        # Deciding for the minkovski p parameter. Have decided for
        # n_neighbors=18 the same way, parametrizing it and checking with CV.
        #clf_constructor = lambda x: KNeighborsClassifier(n_neighbors=10, p=1)
        #k = pick_k_from_CV(range(1,11), clf_constructor, trainData, trainQuality)
        #print k

        clf_constructor = lambda: KNeighborsClassifier(n_neighbors=10, p=1)
        train_predict(clf_constructor)
        print ''
        train_predict_two_clf_one_each_type(clf_constructor)
    if args[0] == 'svm':
        #clf_constructor = lambda k: svm.SVC(gamma=0.001, C=float(k))
        #k = pick_k_from_CV(range(1,201), clf_constructor, trainData, trainQuality)
        #print k
        clf_constructor = lambda: svm.SVC(gamma=0.001, C=100.)
        train_predict(clf_constructor)
        print ''
        train_predict_two_clf_one_each_type(clf_constructor)
    if args[0] == 'logistic_regression':
        # Deciding for the minkovski p parameter. Have decided for
        # n_neighbors=18 the same way, parametrizing it and checking with CV.
        clf_constructor = lambda k: LogisticRegression(C=k)
        k = pick_k_from_CV(np.linspace(0.001,4,10), clf_constructor, trainData, trainQuality)
        print k

        clf_constructor = lambda: LogisticRegression(C=0.8)
        train_predict(clf_constructor)
        print ''
        train_predict_two_clf_one_each_type(clf_constructor)

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
