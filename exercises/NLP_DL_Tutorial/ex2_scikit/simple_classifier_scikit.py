#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'

from readwrite.reader import *
from readwrite.writer import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

def extractFeatures(trainingdata, testdata):
    tweets_train, targets_train, labels_train, ids_train = readTweetsOfficial(trainingdata)
    tweets_test, targets_test, labels_test, ids_test = readTweetsOfficial(testdata)

    features_train, features_test, vocab = featTransform(tweets_train, tweets_test)

    return features_train, labels_train, features_test, labels_test


def featTransform(tweets_train, tweets_test):
    cv = CountVectorizer()
    cv.fit(tweets_train)
    print(cv.vocabulary_)
    features_train = cv.transform(tweets_train)
    features_test = cv.transform(tweets_test)
    return features_train, features_test, cv.vocabulary


def model_train(feats_train, labels):
    # s(f(x), g(x)) + loss function handled by this model
    model = LogisticRegression(penalty='l2')
    model.fit(feats_train, labels)
    return model


def predict(model, features_test):
    """Find the most compatible output class given the input `x` and parameter `theta`"""
    preds = model.predict(features_test)
    #preds_prob = model.predict_proba(features_test)  # probabilities instead of classes
    return preds


if __name__ == '__main__':
    fp = "../data/semeval/"
    train_path = fp + "semeval2016-task6-train+dev.txt"
    test_path = fp + "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    pred_path = fp + "SemEval2016-Task6-subtaskB-testdata-pred.txt"

    features_train, labels_train, features_test, labels_test = extractFeatures(train_path, test_path)
    model = model_train(features_train, labels_train)
    predictions = predict(model, features_test)

    printPredsToFile(test_path, pred_path, predictions)
    eval(test_path, pred_path)