"""
Name: Zach Sahlin
Name: Luke Martin
Class: CPSC 475
Date Submitted: December 13, 2022
Assignment: Project 12
File: utils.py
Description: This program stores the utility functions, such as getting likelihoods
             for Project 12, the Naive Bayes classifier.
To execute: python3 main.py
"""

import math


def read_words(filename):
    """Reads in a file and returns the list of words

    Args:
        filename(str): the name of the file to be read

    Returns:
        list of strs: the words on the file lines
    """
    words = []
    with open(filename, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            words.append(line.strip())

    return words


def word_counts(words):
    """Counts the appearance of each word in a list of words

    Args:
        words(list of strs): a list of words

    Returns:
        counts(dict of str, int): a dictionary containing the count of each word in the list
    """
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

        return counts


def word_likelihoods(words, vocab):
    """Gets the frequency of each word in the vocabularly for a class

    Args:
        words(list of strs): a list of words
        vocab(list of strs): a set of unqiue words

    Returns:
        likelihoods(dict of str, float): a dictionary containing the freq of each word in the vocabulary
    """
    counts = word_counts(words)
    likelihoods = {}
    denominator = len(words) + len(vocab)
    for word in vocab:
        if word in counts:
            likelihoods[word] = (counts[word] + 1) / denominator
        else:
            likelihoods[word] = 1 / denominator
    return likelihoods


def test_review(review, priors, likelihoods):
    """Classify a bag of words constructued from a review using priors and liklihoods

    Args:
        review(list of strs): a list of words
        priors(dict of str, float): class probabilities accessed by class name
        likelihoods(dict of str, dict of str, float): word liklihoods accessed by class name and word

    Returns:
        max_class(str): the predicted classification of the review using Naive Bayes priors/liklihoods
    """
    classes = priors.keys()
 
    sums = {}
    for c in classes:
        sums[c] = math.log(priors[c])
        for word in review:
            if word in likelihoods[c]:
                sums[c] += math.log(likelihoods[c][word])
 
    max_class = max(classes, key= lambda c: sums[c])
    return max_class


def get_confusion_matrix(conf_matrix):
    """Creates a string rep of the confusion matrix along with accuracy, percision, and recall

    Args:
        conf_matrix(dict of str, int): dictionary for storing true/false positives/negatives 

    Returns:
        matrix_string(str): the string representation of conf matrix plus stat measures
    """
    precision = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp'])
    recall = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'])
    accuracy = (conf_matrix['tp'] + conf_matrix['tn']) / (conf_matrix['tp'] + conf_matrix['fp'] + conf_matrix['tn'] + conf_matrix['fn'])
    matrix_string = "Confusion Matrix:\n"
    matrix_string +=  "          | Act. Pos | Act. Neg \n"
    matrix_string += "Pred. Pos | " + str(conf_matrix['tp']) + "       | " + str(conf_matrix['fp']) + "\n"
    matrix_string += "Pred. Neg | " + str(conf_matrix['fn']) + "       | " + str(conf_matrix['tn']) + "\n\n"
    matrix_string += "Accuracy: " + str(round(accuracy, 2)) + "\n"
    matrix_string += "Precision: " + str(round(precision, 2)) + "\n"
    matrix_string += "Recall: " + str(round(recall, 2)) + "\n"

    return matrix_string