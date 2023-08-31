# transform a time series dataset into a supervised learning dataset
import os
import json
import random

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot, pyplot as plt
from numpy import asarray
from pandas import DataFrame, concat

# fit an random forest model and make a one step prediction
from sklearn import metrics, __all__, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, plot_confusion_matrix
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

    RANDOM_SEED = 10

    with open("dictionary.json", "rb") as json_file:
        json_data = json.load(json_file)

    gestures = []
    gesturesTest = []

    labels = np.empty((0,))
    labelsTest = np.empty((0,))

    for root, dirs, files in os.walk(json_data.get('labeledDataTrainSave')):
        random.Random(10).shuffle(files)
        for file in files:
            gestures.append(pd.read_csv(os.path.join(root, file)).values)
            labels = np.append(labels, dictionary[''.join([i for i in file if not i.isdigit()])])

    for root, dirs, files in os.walk(json_data.get('labeledDataTestSave')):
        random.Random(10).shuffle(files)
        for file in files:
            gesturesTest.append(pd.read_csv(os.path.join(root, file)).values)
            labelsTest = np.append(labelsTest, dictionary[''.join([i for i in file if not i.isdigit()])])

    gestures = [item.flatten() for item in gestures]
    gesturesTest = [item.flatten() for item in gesturesTest]

    maxLength = max(gestures, key=lambda array: array.shape[0]).shape[0]
    maxLengthTest = max(gesturesTest, key=lambda array: array.shape[0]).shape[0]

    padded_gestures = []
    padded_gesturesTest = []

    for item in gestures:
        if item.shape[0] < maxLength:
            padded_gestures.append(np.append(item, [0 * x for x in range(maxLength - item.shape[0])]))
        else:
            padded_gestures.append(item)

    for item in gesturesTest:
        if item.shape[0] < maxLength:
            padded_gesturesTest.append(np.append(item, [0 * x for x in range(maxLength - item.shape[0])]))
        else:
            padded_gesturesTest.append(item)

    train = np.reshape(padded_gestures[0], (1, padded_gestures[0].shape[0]))
    test = np.reshape(padded_gesturesTest[0], (1, padded_gesturesTest[0].shape[0]))

    for x in range(1, len(padded_gestures)):
        train = np.append(train, padded_gestures[x].reshape(1, padded_gestures[x].shape[0]), axis=0)

    for x in range(1, len(padded_gesturesTest)):
        test = np.append(test, padded_gesturesTest[x].reshape(1, padded_gesturesTest[x].shape[0]), axis=0)

    randomForest = RandomForestClassifier(n_estimators=1000, random_state=RANDOM_SEED)

    decisionTree = DecisionTreeClassifier(random_state=RANDOM_SEED)

    neighbors = KNeighborsClassifier(n_neighbors=12)

    svmClassifier = svm.SVC(kernel='linear', random_state=RANDOM_SEED)

    gaussian = GaussianNB()

    gaussianMixture = GaussianMixture(random_state=RANDOM_SEED)

    randomForest.fit(train, labels)

    decisionTree.fit(train, labels)

    gaussian.fit(train, labels)

    gaussianMixture.fit(train, labels)

    neighbors.fit(train, labels)

    svmClassifier.fit(train, labels)

    pred = randomForest.predict(test)
    predNeighbors = neighbors.predict(test)
    predGaussian = gaussian.predict(test)
    predGaussianMixture = gaussianMixture.predict(test)
    predSVM = svmClassifier.predict(test)
    predDecisionTree = decisionTree.predict(test)

    print("Random Forest:")
    print("Accuracy:", metrics.accuracy_score(labelsTest, pred))
    print("F1-score:", metrics.f1_score(labelsTest, pred,average=None))

    plt.figure(figsize=(250, 250))
    plot_confusion_matrix(randomForest, test, labelsTest, normalize="true")
    plt.savefig("RandomForest_All_Gestures.png")
    print("")
    print("K-Nearest-Neighbors:")
    print("Accuracy:", metrics.accuracy_score(labelsTest, predNeighbors))
    print("F1-score:", metrics.f1_score(labelsTest, predNeighbors, average=None))

    plot_confusion_matrix(neighbors, test, labelsTest, normalize="true")
    plt.savefig("K-Nearest-Neighbors_All_Gestures.png")
    print("")
    print("Naive Bayes:")
    print("Accuracy:", metrics.accuracy_score(labelsTest, predGaussian))
    print("F1-score:", metrics.f1_score(labelsTest, predGaussian, average=None))

    plot_confusion_matrix(gaussian, test, labelsTest, normalize="true")
    plt.savefig("Naive_Bayes_All_Gestures.png")
    print("")
    print("GaussianMixture:")
    print("Accuracy:", metrics.accuracy_score(labelsTest, predGaussianMixture))
    print("F1-score:", metrics.f1_score(labelsTest, predGaussianMixture, average=None))

    #plot_confusion_matrix(gaussianMixture, test, labelsTest)
    #plt.savefig("Gaussian_Mixture_All_Gestures.png")

    print("")
    print("Support Vector Machine:")
    print("Accuracy:", metrics.accuracy_score(labelsTest, predSVM))
    print("F1-score:", metrics.f1_score(labelsTest, predSVM, average=None))

    plot_confusion_matrix(svmClassifier, test, labelsTest, normalize="true")
    plt.savefig("Support_Vector_Machine_All_Gestures.png")

    print("")
    print("Decision Tree:")
    print("Accuracy:", metrics.accuracy_score(labelsTest, predDecisionTree))
    print("F1-score:", metrics.f1_score(labelsTest, predDecisionTree, average=None))

    plot_confusion_matrix(decisionTree, test, labelsTest, normalize="true")
    plt.savefig("Decision_Tree_All_Gestures.png")
