# transform a time series dataset into a supervised learning dataset
import os
import json

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot
from numpy import asarray
from pandas import DataFrame, concat

# fit an random forest model and make a one step prediction
from sklearn import metrics, __all__, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

dictionary = {"TILT_HEAD_LEFT.csv": 0, "TILT_HEAD_RIGHT.csv": 1, "TAP_GLASSES_LEFT.csv": 2,
              "TAP_GLASSES_RIGHT.csv": 3, "SLOW_NOD.csv": 4, "PUSH_GLASSES_UP.csv": 5,
              "READJUST_GLASSES_LEFT.csv": 6, "READJUST_GLASSES_RIGHT.csv": 7,
              "TAP_NOSE_LEFT.csv": 8, "TAP_NOSE_RIGHT.csv": 9, "RUB_NOSE.csv": 10}

# split a univariate dataset into train/test sets
def train_test_split(data, labels,  n_test):
    return data[:-n_test, :], labels[:-n_test], data[-n_test:, :], labels[-n_test:]


if __name__ == '__main__':

    with open("dictionary.json", "rb") as json_file:
        json_data = json.load(json_file)

    gestures = []
    labels = np.empty((0,))

    for root, dirs, files in os.walk(json_data.get('labeledDataSave')):
        for file in files:
            gestures.append(pd.read_csv(os.path.join(root, file)).values)
            labels = np.append(labels, dictionary[''.join([i for i in file if not i.isdigit()])])

    gestures = [item.flatten() for item in gestures]

    maxLength = max(gestures, key=lambda array: array.shape[0]).shape[0]

    padded_gestures = []

    for item in gestures:
        if item.shape[0] < maxLength:
            padded_gestures.append(np.append(item, [0 * x for x in range(maxLength - item.shape[0])]))
        else:
            padded_gestures.append(item)

    padded = np.reshape(padded_gestures[0], (1, padded_gestures[0].shape[0]))
    for x in range(1, len(padded_gestures)):
        padded = np.append(padded, padded_gestures[x].reshape(1, padded_gestures[x].shape[0]), axis=0)

    randomForest = RandomForestClassifier(n_estimators=1000)

    decisionTree = DecisionTreeClassifier()

    neighbors = KNeighborsClassifier(n_neighbors=3)

    svmClassifier = svm.SVC(kernel='linear')

    gaussian = GaussianNB()

    gaussianMixture = GaussianMixture()

    train, train_labels, test, test_labels = train_test_split(padded, labels, 132)

    randomForest.fit(train, train_labels)

    decisionTree.fit(train, train_labels)

    gaussian.fit(train, train_labels)

    gaussianMixture.fit(train, train_labels)

    neighbors.fit(train, train_labels)

    svmClassifier.fit(train, train_labels)

    pred = randomForest.predict(test)
    predNeighbors = neighbors.predict(test)
    predGaussian = gaussian.predict(test)
    predGaussianMixture = gaussianMixture.predict(test)
    predSVM = svmClassifier.predict(test)
    predDecisionTree = decisionTree.predict(test)

    print("Random Forest:")
    print("Accuracy:", metrics.accuracy_score(test_labels, pred))
    print("F1-score:", metrics.f1_score(test_labels, pred,average=None))
    print("")
    print("K-Nearest-Neighbors:")
    print("Accuracy:", metrics.accuracy_score(test_labels, predNeighbors))
    print("F1-score:", metrics.f1_score(test_labels, predNeighbors, average=None))
    print("")
    print("Naive Bayes:")
    print("Accuracy:", metrics.accuracy_score(test_labels, predGaussian))
    print("F1-score:", metrics.f1_score(test_labels, predGaussian, average=None))
    print("")
    print("GaussianMixture:")
    print("Accuracy:", metrics.accuracy_score(test_labels, predGaussianMixture))
    print("F1-score:", metrics.f1_score(test_labels, predGaussianMixture, average=None))
    print("")
    print("Support Vector Machine:")
    print("Accuracy:", metrics.accuracy_score(test_labels, predSVM))
    print("F1-score:", metrics.f1_score(test_labels, predSVM, average=None))
    print("")
    print("Decision Tree:")
    print("Accuracy:", metrics.accuracy_score(test_labels, predDecisionTree))
    print("F1-score:", metrics.f1_score(test_labels, predDecisionTree, average=None))
