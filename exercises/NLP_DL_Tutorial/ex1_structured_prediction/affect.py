#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'

import numpy as np


def getAffect(tweets):

    # Impact of gaz features
    # affect_anger.lst  -
    # affect_bad.lst o
    # affect_disgust.lst -
    # affect_fear.lst +
    # affect_joy.lst -
    # affect_sadness.lst +
    # affect_surprise.lst o

    #files = ["affect_fear.lst", "affect_surprise.lst", "affect_bad.lst", "affect_sadness.lst"]
    #vocab = ["fear", "surprise", "bad", "sadness"]
    files = ["affect_anger.lst", "affect_bad.lst", "affect_disgust.lst", "affect_fear.lst", "affect_joy.lst", "affect_sadness.lst",
             "affect_surprise.lst"]
    vocab = ["anger", "bad", "disgust", "fear", "joy", "sadness", "surprise"]
    vects = []
    gaz = []

    for f in files:
        ga = []
        for line in open("../data/gazetteers/" + f, 'r'):
            ga.append(line.split("&")[0])
        gaz.append(ga)

    for tweet in tweets:
        vect = np.zeros(len(vocab))
        for i, g in enumerate(gaz):
            affect = 0
            for entry in g:
                if entry in tweet: # note that we use a very simple heuristic and don't check token or phrase boundaries here
                    affect += 1  # small training set, probably doesn't make sense to introduce counts for that
                    #break
            vect[i] = affect
        vects.append(vect)

    return vects, vocab



def getAffectCounts(tweets, labels, train=True):

    # Impact of gaz features
    # affect_anger.lst  -
    # affect_bad.lst o
    # affect_disgust.lst -
    # affect_fear.lst +
    # affect_joy.lst -
    # affect_sadness.lst +
    # affect_surprise.lst o

    #files = ["affect_anger.lst", "affect_bad.lst", "affect_disgust.lst", "affect_fear.lst", "affect_joy.lst", "affect_sadness.lst",
    #         "affect_surprise.lst"]

    files = ["affect_fear.lst", "affect_surprise.lst", "affect_bad.lst", "affect_sadness.lst"]

    gaz_pos = set()
    gaz_neg = set()
    counts_p_n = []

    for f in files:
        for line in open("../data/gazetteers/" + f, 'r'):
            if f == "affect_joy.lst" or f == "affect_surprise.lst":
                gaz_pos.add(line.split("&")[0])
            else:
                gaz_neg.add(line.split("&")[0])

    for i, tweet in enumerate(tweets):

        # binary classification for now
        if train == True and labels[i] == "NONE":
            continue

        lab = -1
        if labels[i] == "FAVOR":
            lab = 1

        affect_pos = 0
        affect_neg = 0
        for ge in gaz_pos:
            if ge in tweet:
                affect_pos += 1
        for ge in gaz_neg:
            if ge in tweet:
                affect_neg += 1

        counts_p_n.append([affect_pos-affect_neg, lab])


    return counts_p_n


if __name__ == '__main__':
    pass