#!/usr/bin/env python3

__author__ = 'Isabelle Augenstein'


from gensim.models import word2vec, Phrases
from ex3_word2vec.twokenize_wrapper import tokenize
from ex3_word2vec.tokenize_tweets import readTweets, filterStopwords
import logging
from readwrite.reader import readTweetsOfficial



# prep data for word2vec
def prepData(filepath, stopfilter, multiword):
    print("Preparing data...")

    ret = [] # list of lists

    print("Reading data...")
    # this reads file in JSON format
    #tweets = readTweets(jsonfilepath)

    # this reads SemEval format tweets
    tweets, _, _, _ = readTweetsOfficial(filepath)
    #tweets = "\n".join(tweets)

    print("Tokenising...")
    for tweet in tweets:
        tokenised_tweet = tokenize(tweet.lower())
        if stopfilter:
            words = filterStopwords(tokenised_tweet)
            ret.append(words)
        else:
            ret.append(tokenised_tweet)

    if multiword:
        return learnMultiword(ret)
    else:
        return ret


def learnMultiword(ret, outpath="phrase_all.model"):
    print("Learning multiword expressions")
    bigram = Phrases(ret)
    bigram.save(outpath)

    print("Sanity checking multiword expressions")
    test = "i like donald trump and hate muslims , go hillary , i like jesus , jesus , against , abortion "
    sent = test.split(" ")
    print(bigram[sent])
    return bigram[ret]



def trainWord2VecModel(jsonfilepath, stopfilter, multiword, modelname):
    tweets = prepData(jsonfilepath, stopfilter, multiword)
    print("Starting word2vec training")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # set params
    num_features = 100    # Word vector dimensionality
    min_word_count = 2   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    trainalgo = 1 # cbow: 0 / skip-gram: 1

    print("Training model...")
    model = word2vec.Word2Vec(tweets, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, sg = trainalgo)

    # add for memory efficiency
    model.init_sims(replace=True)

    # save the model
    model.save(modelname)



# find most similar n words to given word
def applyWord2VecMostSimilar(modelname="skip_nostop_multi_100features_10minwords_10context", word="#abortion", top=20):
    model = word2vec.Word2Vec.load(modelname)
    print("Find ", top, " terms most similar to ", word, "...")
    for res in model.wv.most_similar(word, topn=top):
        print(res)
    print("\n")


# determine similarity between words
def applyWord2VecSimilarityBetweenWords(modelname="skip_nostop_multi_100features_10minwords_10context", w1="trump", w2="muslims"):
    model = word2vec.Word2Vec.load(modelname)
    print("Computing similarity between ", w1, " and ", w2, "...")
    print(model.wv.similarity(w1, w2), "\n")


# search which words/phrases the model knows which contain a searchterm
def applyWord2VecFindWord(modelname="skip_nostop_multi_100features_10minwords_10context", searchterm="trump"):
    model = word2vec.Word2Vec.load(modelname)
    print("Finding terms containing ", searchterm, "...")
    for v in model.vocab:
        if searchterm in v:
            print(v.encode('utf-8'))
    print("\n")


if __name__ == '__main__':

    fp = "../data/semeval/"
    train_path = fp + "semeval2016-task6-train+dev.txt"
    test_path = fp + "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    pred_path = fp + "SemEval2016-Task6-subtaskB-testdata-pred.txt"

    #trainWord2VecModel(train_path, stopfilter=True, multiword=False, modelname="skip_nostop_sing_100features_5minwords_10context")

    # Below: simple word2vec test for a trained model with phrases recognised as part of preprocessing

    applyWord2VecMostSimilar("skip_nostop_sing_100features_5minwords_10context", "#makeamericagreatagain", 20)
    applyWord2VecSimilarityBetweenWords("skip_nostop_sing_100features_5minwords_10context", "#makeamericagreatagain", "trump")
    applyWord2VecFindWord("skip_nostop_sing_100features_5minwords_10context", "trump")

