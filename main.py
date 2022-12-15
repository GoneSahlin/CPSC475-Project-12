"""
Name: Zach Sahlin
Name: Luke Martin
Class: CPSC 475
Date Submitted: December 13, 2022
Assignment: Project 12
File: main.py
Description: This program trains a Naive Bayes classifier on both positive and
             negative movie reviews, and then tests it using a testing set. The movie
             reviews are obtained from the NLTK library.
To execute: python3 main.py
"""

from utils import *
from makeData import getBagOfWords


def main():
    """
    Main function
    """
    # Training
    classes = ["neg", "pos"]
    priors = {"neg": 0.5, "pos": 0.5}
 
    pos_words_train = read_words('pos.txt')
    neg_words_train = read_words('neg.txt')
    V = set(pos_words_train + neg_words_train)
   
    pos_likelihoods = word_likelihoods(pos_words_train, V)
    neg_likelihoods = word_likelihoods(neg_words_train, V)
    likelihoods = {"pos": pos_likelihoods, "neg": neg_likelihoods}

    # Testing
    pos_reviews = read_words('posTst.txt')
    neg_reviews = read_words('negTst.txt')

    conf_matrix = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0 }

    for pos_review in pos_reviews:
        review_words = getBagOfWords(pos_review)
        prediction_class = test_review(review_words, priors, likelihoods)

        if prediction_class == "pos":
            conf_matrix['tp'] += 1
        else:
            conf_matrix['fn'] += 1

    for neg_review in neg_reviews:
        review_words = getBagOfWords(neg_review)
        prediction_class = test_review(review_words, priors, likelihoods)

        if prediction_class == "neg":
            conf_matrix['tn'] += 1
        else:
            conf_matrix['fp'] += 1

    confusion_matrix = get_confusion_matrix(conf_matrix)

    print(confusion_matrix)


if __name__ == '__main__':
    main()
