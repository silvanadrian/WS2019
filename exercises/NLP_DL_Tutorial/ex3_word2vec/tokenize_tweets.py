#!/usr/bin/env python3

import json
from ex3_word2vec.twokenize_wrapper import tokenize


# read tweets from json, get numbers corresponding to tokens from file
def readTweets(jsonfilepath):
    tweets = []
    for line in open(jsonfilepath, 'r'):
        tweets.append(json.loads(line)['text'])
    return tweets

def tokenise_tweets(tweets):
    return [filterStopwords(tokenize(tweet.lower())) for tweet in tweets]


def filterStopwords(tokenised_tweet):

    # this seems to work best for conditional LSTM training
    stops = ["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":", ";", "<", ">", "@",
             "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?", "rt", "#semst", "...", "thats", "im", "'s", "via"]
    return [w for w in tokenised_tweet if (not w in stops and not w.startswith("http"))]




if __name__=='__main__':
    input = ["i", "like", "tokenisation", "!"]
    print(input)
    print(filterStopwords(input))