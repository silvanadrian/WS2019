#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'


from gensim.models import word2vec, Phrases
from ex3_word2vec.tokenize_tweets import tokenise_tweets
from sklearn.linear_model import LogisticRegression
from readwrite.reader import *
from readwrite.writer import *



def extractFeaturesW2V(trainingdata, testdata, w2vpath="GoogleNews-vectors-negative300.bin", gnews=True, dim=300, usePhrase=False, phrasemodelpath="phrase_all.model", cross_features="true"):

    if usePhrase == True:
        phmodel = Phrases.load(phrasemodelpath)

    tweets_train, targets_train, labels_train, ids_train = readTweetsOfficial(trainingdata)

    tweet_tokens = tokenise_tweets(tweets_train)
    target_tokens = tokenise_tweets(targets_train)

    if usePhrase == True:
        tweet_tokens = phmodel[tweet_tokens]
        target_tokens = phmodel[target_tokens]

    tweets_test, targets_test, labels_test, ids_test = readTweetsOfficial(testdata)

    tweet_tokens_test = tokenise_tweets(tweets_test)
    target_tokens_test = tokenise_tweets(targets_test)

    if usePhrase == True:
        tweet_tokens_test = phmodel[tweet_tokens_test]
        target_tokens_test = phmodel[target_tokens_test]

    if gnews == True:
        w2vmodel = word2vec.Word2Vec.load_word2vec_format(w2vpath, binary=True)
    else:
        w2vmodel = word2vec.Word2Vec.load(w2vpath)


    features_train_w2v_tweet = encodeSentW2V(w2vmodel, tweet_tokens, dim)
    features_train_w2v_targ = encodeSentW2V(w2vmodel, target_tokens, dim)

    features_dev_w2v_tweet = encodeSentW2V(w2vmodel, tweet_tokens_test, dim)
    features_dev_w2v_target = encodeSentW2V(w2vmodel, target_tokens_test, dim)

    features_train_w2v = extrFeaturesW2V(features_train_w2v_tweet, features_train_w2v_targ, cross_features)
    features_dev_w2v = extrFeaturesW2V(features_dev_w2v_tweet, features_dev_w2v_target, cross_features)

    return features_train_w2v, labels_train, features_dev_w2v, labels_test


def extrFeaturesW2V(encoded_tweet, encoded_target, cross_features="true"):
    """
    :param cross_features: "true": outher product of tweet and target representations; "added": concat tweet and target
    representations; "tweetonly": only tweet representations
    :return:
    """
    features_train = []
    if cross_features == "true":
        for i, enc in enumerate(encoded_target):
            features_train_i = []
            for v in np.outer(encoded_tweet[i], encoded_target[i]):
                features_train_i.extend(v)
            features_train.append(features_train_i)
    elif cross_features == "added":
        for i, enc in enumerate(encoded_target):
            features_train.append(np.append(encoded_tweet[i], enc))
    else:
        features_train = encoded_tweet

    print("Features extracted!")

    return features_train


def encodeSentW2V(w2vmodel, sents, dim):

    feats = []
    # for each tweet, get the word vectors and average them
    for i, tweet in enumerate(sents):
        numvects = 0
        vect = []
        for token in tweet:
            try:
                s = w2vmodel.wv[token]
                vect.append(s)
                numvects += 1
            except KeyError:
                s = 0.0
        if vect.__len__() > 0:
            mtrmean = np.average(vect, axis=0)
            if i == 0:
                feats = mtrmean
            else:
                feats = np.vstack((feats, mtrmean))
        else:
            feats = np.vstack((feats, np.zeros(dim)))

    return feats




# train one three-way classifier
def train_classifier_3way_simp(feats_train, labels_train, feats_dev, labels_dev):
    labels = []  # -1 for NONE, 0 for AGAINST, 1 for FAVOR
    labels_dev_tr = [] #transformed from "NONE" etc to -1,0,1

    for i, lab in enumerate(labels_train):
        if lab == 'NONE':
            labels.append(0)
        elif lab == 'FAVOR':
            labels.append(2)
        elif lab == 'AGAINST':
            labels.append(1)

    for i, lab in enumerate(labels_dev):
        if lab == 'NONE' or lab == 'UNKNOWN':
            labels_dev_tr.append(0)
        elif lab == 'FAVOR':
            labels_dev_tr.append(2)
        elif lab == 'AGAINST':
            labels_dev_tr.append(1)


    print("Training classifier...")

    model = LogisticRegression(penalty='l2')#, class_weight='balanced') #svm.SVC(class_weight={1: weight})
    model.fit(feats_train, labels)
    preds = model.predict(feats_dev)
    preds_prob = model.predict_proba(feats_dev)
    coef = model.coef_
    print("Label options", model.classes_)

    print("Labels", labels_dev_tr)
    print("Predictions", preds)
    print("Predictions prob", preds_prob)
    print("Feat length ", feats_train[0].__len__())


    return preds



if __name__ == '__main__':

    fp = "../data/semeval/"
    train_path = fp + "semeval2016-task6-train+dev.txt"
    test_path = fp + "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    pred_path = fp + "SemEval2016-Task6-subtaskB-testdata-pred.txt"

    # get vec for every word/seq, concatenate
    features_train, labels_train, features_dev, labels_dev = extractFeaturesW2V(train_path, test_path, "skip_nostop_sing_100features_5minwords_10context",
                                                                                gnews = False, dim=100, usePhrase=False, phrasemodelpath="phrase_all.model", cross_features="added")

    # train_classifier_3waySGD is another option, for testing elastic net regularisation, doesn't work as well as just l2 though
    preds = train_classifier_3way_simp(features_train, labels_train, features_dev, labels_dev)

    printPredsToFile(test_path, pred_path, preds)

    eval(test_path, pred_path)