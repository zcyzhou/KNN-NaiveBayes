"""
This file includes some utility functions
"""
import csv
import numpy as np


def load_data(file_name, file_type):
    """
    Loading the dataset and return corresponding attributes and labels

    :param file_type: clarity if the file is for testing or training
    :param file_name: target dataset
    :return: attributes, labels as list
    """
    attributes = []
    labels = []
    with open(file_name) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if file_type == "train":
                attributes.append(np.array(row[0:-1]).astype(float))
                labels.append(row[-1])
            else:
                attributes.append(np.array(row).astype(float))

    return attributes, labels


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))


def normal_distribution_pdf(x, mean, sd):
    return (1/(sd * np.sqrt(2*np.pi))) * np.exp(-(x-mean) ** 2/(2 * (sd ** 2)))
