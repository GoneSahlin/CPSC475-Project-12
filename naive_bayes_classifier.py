import numpy as np

from mypytable import MyPyTable

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict, key is y_label): The prior probabilities computed for each
            label in the training set.
        posteriors(3x nested dict, 1st key is att, 2nd key is x_label, 3rd key is y_label): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # priors
        y_labels = []
        for y in y_train:
            if y not in y_labels:
                y_labels.append(y)

        self.priors = {}
        for y_label in y_labels:
            self.priors[y_label] = sum([1 if y == y_label else 0 for y in y_train]) / len(y_train)

        #posteriors
        X_train_table = MyPyTable(list(range(len(X_train[0]))), X_train)
        self.posteriors = {}
        for att in X_train_table.column_names:
            col = X_train_table.get_column(att)
            # col = [X_train_table.data[i][att] for i in range(len(X_train_table.data))]
            col_labels = []
            for x in col:
                if x not in col_labels:
                    col_labels.append(x)

            self.posteriors[att] = {}

            for x_label in col_labels:
                self.posteriors[att][x_label] = {}

                for y_label in y_labels:
                    correct_y_indexes = []
                    for i, y in enumerate(y_train):
                        if y_label == y:
                            correct_y_indexes.append(i)

                    count = sum([1 if x_label == col[index] else 0 for index in correct_y_indexes])

                    self.posteriors[att][x_label][y_label] = count / len(correct_y_indexes)



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in X_test:
            y_labels = []
            probs = []
            for y_label in self.priors:
                y_labels.append(y_label)

                prob = 1
                for i, x_label in enumerate(x):
                    if x_label in self.posteriors[i]:
                        prob *= self.posteriors[i][x_label][y_label]

                prob *= self.priors[y_label]
                probs.append(prob)

            max_prob = max(probs)
            max_indices = []
            for i, prob in enumerate(probs):
                if prob == max_prob:
                    max_indices.append(i)

            if len(max_indices) == 1:
                y_predicted.append(y_labels[max_indices[0]])
            else:
                index = np.random.choice(max_indices)
                y_predicted.append(y_labels[index])

        return y_predicted
