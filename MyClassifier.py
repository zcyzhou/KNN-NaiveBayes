"""
This is the main program of the implementation of KNN and Naive Bayes algorithms
"""

import sys

from classifiers.utils import load_data
from classifiers.knn import KNN

TRAINING_FILE = sys.argv[1]
TESTING_FILE = sys.argv[2]
CLASSIFIER = sys.argv[3]


# Load the training and testing dataset
x_train, y_train = load_data(TRAINING_FILE)
x_test, y_test = load_data(TESTING_FILE)

if __name__ == '__main__':
    if CLASSIFIER == "NB":
        pass
    else:
        k = int(CLASSIFIER[0])
        classifier = KNN(k, x_train, y_train)
        result = []
        for i in x_test:
            result.append(classifier.knn(i))
        for i in result:
            print(i)
        # same_num = 0
        # for i, j in zip(result, y_test):
        #     if i == j:
        #         same_num += 1
        # print(same_num/len(result))

