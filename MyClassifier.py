"""
This is the main program of the implementation of KNN and Naive Bayes algorithms
"""

import sys

from classifiers.utils import load_data
from classifiers.knn import KNN
from classifiers.naivebayes import NaiveBayes

TRAINING_FILE = sys.argv[1]
TESTING_FILE = sys.argv[2]
CLASSIFIER = sys.argv[3]


# Load the training and testing dataset
x_train, y_train = load_data(TRAINING_FILE, "train")
x_test, y_test = load_data(TESTING_FILE, "test")

if __name__ == '__main__':
    if CLASSIFIER == "NB":
        classifiers = NaiveBayes(x_train, y_train)
        classifiers.init_processing()
        result = []
        for i in x_test:
            print(classifiers.nb(i))

    else:
        k = int(CLASSIFIER[0])
        classifier = KNN(k, x_train, y_train)
        result = []
        for i in x_test:
            print(classifier.knn(i))

