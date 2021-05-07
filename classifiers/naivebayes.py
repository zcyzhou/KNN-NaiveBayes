import numpy as np

from classifiers.utils import normal_distribution_pdf


class NaiveBayes:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.yes_p = 0
        self.no_p = 0
        # The table is in format: [[sd, mean], [], [], []]
        self.yes_table = np.zeros((len(self.x_train[0]), len(np.unique(self.y_train))))
        self.no_table = np.zeros((len(self.x_train[0]), len(np.unique(self.y_train))))

    def init_processing(self):
        # Separate data into two set according to the label
        yes_indices = []
        no_indices = []
        for i in range(len(self.y_train)):
            if self.y_train[i] == "yes":
                yes_indices.append(i)
            else:
                no_indices.append(i)
        self.yes_p = len(yes_indices)/len(self.y_train)
        self.no_p = len(no_indices)/len(self.y_train)
        yes_x_train = [self.x_train[i] for i in yes_indices]
        no_x_train = [self.x_train[i] for i in no_indices]
        yes_x_train = np.transpose(yes_x_train)
        no_x_train = np.transpose(no_x_train)

        for i in range(len(yes_x_train)):
            self.yes_table[i][0] = np.mean(yes_x_train[i])
            self.yes_table[i][1] = np.sqrt(np.sum((yes_x_train[i] - self.yes_table[i][0])**2)/(len(yes_x_train[i])-1))
            self.no_table[i][0] = np.mean(no_x_train[i])
            self.no_table[i][1] = np.sqrt(np.sum((no_x_train[i] - self.no_table[i][0])**2)/(len(no_x_train[i])-1))

    def nb(self, x):
        yes_prob = 1
        no_prob = 1
        for i in range(len(x)):
            yes_prob = normal_distribution_pdf(x[i], self.yes_table[i][0], self.yes_table[i][1]) * yes_prob
            no_prob = normal_distribution_pdf(x[i], self.no_table[i][0], self.no_table[i][1]) * no_prob
        yes_prob *= self.yes_p
        no_prob *= self.no_p

        return "yes" if yes_prob/no_prob >= 1 else "no"
