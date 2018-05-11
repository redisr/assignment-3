import numpy as np
import pandas as pd
import re

import nltk.data

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from gensim.models import word2vec
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud


def headline_to_wordlist( headline, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    headline_text = BeautifulSoup(headline, "lxml").get_text()
    #
    # 2. Remove non-letters
    headline_text = re.sub("[^a-zA-Z]"," ", headline_text)
    #
    # 3. Convert words to lower case and split them
    words = headline.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


def headline_to_sentence(headline, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words

    #Please, download the punkt tokenizer from NLTK
    #If it is already downloaded, just close the downloader window

    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(headline.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( headline_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def get_sentences(X):
    sentences = []
    # for iter in range(X.shape[0]):
    for iter in range(10000):
    # for iter in range(10):
        sentences += headline_to_sentence(X[iter][1])

    return sentences

# def pattern2vector(tokens, word2vec, AVG=False):
#     pattern_vector = np.zeros(word2vec.layer1_size)
#     n_words = 0
#     if len(tokens) > 1:
#         for t in tokens:
#             try:
#                 vector = word2vec[t.strip()]
#                 pattern_vector = np.add(pattern_vector,vector)
#                 n_words += 1
#             except KeyError, e:
#                 continue
#         if AVG is True:
#             pattern_vector = np.divide(pattern_vector,n_words)
#     elif len(tokens) == 1:
#         try:
#             pattern_vector = word2vec[tokens[0].strip()]
#         except KeyError:
#             pass
#     return pattern_vector


def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.vocab]
    return np.mean(word2vec_model[doc], axis=0)

############################################################################################################

if __name__ == '__main__':
    nltk.download()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    dataset = pd.read_csv('news_headlines/news_headlines.csv')
    X = dataset.iloc[1:, :].values

    print("Parsing sentences...")
    sentences = get_sentences(X)

    #Train word2vec model
    print("Training Word2Vec model...")
    model = word2vec.Word2Vec(sentences, workers = 4, size = 1000, min_count = 30, window = 10, sample = 0.001)

    print("Getting average vector for each sentence...")
    # word_vectors = []
    # for sentence in sentences:
    #     word_vectors.append(document_vector(model, sentence))
    #
    # print(len(word_vectors))

    # #Get Inputs to train K-means
    # word_vectors = model.wv.syn0

    # for n_clusters in range(3,20):
    #     print("Training K-Means for n_clusters = " + str(n_clusters))
    #     kmeans = KMeans(n_clusters=int(n_clusters), init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=-1)
    #     cluster_labels = kmeans.fit_predict(word_vectors)
    #
    #     silhouette_avg = silhouette_score(word_vectors, cluster_labels, sample_size=1000)
    #
    #     #High silhouette values mean that the clusters are well separated.
    #     print("For n_clusters =", n_clusters,
    #           "The average silhouette_score is :", silhouette_avg)
