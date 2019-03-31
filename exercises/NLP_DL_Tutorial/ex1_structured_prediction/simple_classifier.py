#import sys
#sys.path.append("/path/to/BridgesMLTutorial")

#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'


from readwrite.reader import *
from readwrite.writer import *
from ex1_structured_prediction.affect import *
import math
from random import randint

def extractFeatures(trainingdata, testdata):
    tweets_train, targets_train, labels_train, ids_train = readTweetsOfficial(trainingdata)
    counts_p_n_train = getAffectCounts(tweets_train, labels_train)
    tweets_test, targets_test, labels_test, ids_test = readTweetsOfficial(testdata)
    counts_p_n_test = getAffectCounts(tweets_test, labels_test, False)

    return counts_p_n_train, labels_train, counts_p_n_test, labels_test


def s(theta, f_x, f_y):
    """Measure the compatibility of x and y instances using parameter `theta`"""
    if (f_x > 0 and f_y > 0) or (f_x < 0 and f_y < 0):  # if both pos or both neg, high compatibility
        comp = 1
    else:
        comp = 0
    return theta * comp


def loss(theta, data, vocab):
    """Measure the total number of errors made when predicting with parameter `theta` on training set `data`"""
    total = 0.0
    for x,y in data:
        max_score = -math.inf
        scores = set()
        result = None
        for y_guess in vocab:
            score = s(theta,x,y_guess)
            if score > max_score:
                result = y_guess
                max_score = score
            scores.add(score)
        # check if there is a difference between the choices
        # if so, take best guess and add 1 to total loss if there's a mismatch
        if len(scores) > 1 and result != y:
            total += 1.0
    return total


def train(train, vocab):
    l1 = loss(1.0, train, vocab)
    l_1 = loss(-1.0, train, vocab)
    theta_star = 1.0 if l1 < l_1 else -1.0
    return theta_star


def predict(theta, x, y_space):
    """Find the most compatible output class given the input `x` and parameter `theta`"""
    max_score = -math.inf
    scores = set()
    result = None
    for y_guess in y_space:
        score = s(theta, x, y_guess)
        if score > max_score:
            result = y_guess
            max_score = score
        scores.add(score)
    #if it doesn't make a difference, pick a random one
    if len(scores) == 1:
        result = randint(0, len(y_space)-1)
        result = y_space[result]
    return result


if __name__ == '__main__':

    fp = "../data/semeval/"
    train_path = fp + "semeval2016-task6-train+dev.txt"
    test_path = fp + "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    pred_path = fp + "SemEval2016-Task6-subtaskB-testdata-pred.txt"

    vocab = [1, -1]

    counts_p_n_train, labels_train, counts_p_n_test, labels_test = extractFeatures(train_path, test_path)

    theta_star = train(counts_p_n_train, vocab)

    predictions = []
    for x, y in counts_p_n_test:
        pred = predict(theta_star, x, vocab)
        if pred == 1: # FAVOR
            pred = 2
        else:  # AGAINST
            pred = 1

        predictions.append(pred)


    printPredsToFile(test_path, pred_path, predictions)
    eval(test_path, pred_path)