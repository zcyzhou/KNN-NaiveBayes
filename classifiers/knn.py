import numpy as np

from classifiers.utils import euclidean_distance


class KNN:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def knn(self, new_x):
        kn_idx = np.argsort([euclidean_distance(new_x, x) for x in self.x_train])[0:self.k]
        kn_y = [self.y_train[idx] for idx in kn_idx]
        yes_count = 0
        no_count = 0
        for y in kn_y:
            if y == "yes":
                yes_count += 1
            else:
                no_count += 1

        return "yes" if yes_count >= no_count else "no"

