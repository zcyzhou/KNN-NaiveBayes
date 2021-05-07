"""
This file includes some utility functions
"""
import numpy as np


def load_data(file_name):
    """
    Loading the dataset and return corresponding attributes and labels

    :param file_name: target dataset
    :return: attributes, labels as list
    """
    attributes = []
    labels = []
    with open(file_name) as f:
        data = f.readlines()
        for line in data:
            temp = line.strip('\n').split(',')
            attributes.append(np.array(temp[0:-1]).astype(float))
            labels.append(temp[-1])

    return attributes, labels


def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a-b)**2))