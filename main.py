import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from utils import get_texts_by_year, get_years, lemmatize_data

def get_sentences(X, tokenizer):
    sentences = []
    for iter in range(X.shape[0]):
        print("sentence: " + str(iter))
        # sentences += headline_to_sentence(X[iter][1], tokenizer)
        sentences.append(X[iter][1])

    print(sentences)
    return np.asarray(sentences)

if __name__ == '__main__':
    dataset = pd.read_csv('news_headlines/news_headlines.csv')
    X = dataset.iloc[1:, :].values
    # years = [2003, 2004]

    #Train with all sentences to check what's going on
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("Get sentences...")
    data = get_sentences(X, tokenizer)
    # for i in years:
    # print (i)
    # data = get_texts_by_year(X, i)

    #TODO: I'll lemmatize the data later
    # print("Lemmatize data")
    # print(data)
    # data = lemmatize_data(data)
    # print(data[0])
    # input()
    # vectorize the sentences to ngram (2-grams and 3-grams)
    print("Vectorize data")
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=100)
    tokens = vectorizer.fit_transform(data)

    # Run this and go watch some netflix, get some sleep, and check it tomorrow
    wcss = []
    print("K-Means times")
    for clusters in range(3,20):
        print (clusters)
        kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0, n_jobs=-1)
        y_labels = kmeans.fit_predict(tokens)
        silhouette_avg = silhouette_score(tokens, y_labels, sample_size=1000)

        #High silhouette values mean that the clusters are well separated.
        print("For n_clusters =", clusters,
              "The average silhouette_score is :", silhouette_avg)
        wcss.append(kmeans.inertia_)

    # plot cost function
    plt.plot(range(3, 20), wcss)
    plt.savefig("graphs/cost-" + str(i) + "(2, 2)")
    plt.show()
        # n_clusters = input()

        # # Find clusters
        # kmeans = KMeans(n_clusters=int(n_clusters), init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=3)
        # y_kmeans = kmeans.fit_predict(tokens)
        #
        # sorted_indexes = np.argsort(y_kmeans)
        # texts = dict()
        #
        # # separate clusters
        # clusters = dict()
        # for indexes in sorted_indexes:
        #     if not(y_kmeans[indexes] in clusters):
        #         print (y_kmeans[indexes])
        #         clusters[y_kmeans[indexes]] = []
        #     clusters[y_kmeans[indexes]].append(indexes)
        #
        #
        # # make word clouds out of each cluster
        # wordclouds = []
        # for idx in clusters:
        #     wordclouds.append(WordCloud().generate(" ".join(X[clusters[idx], 1])))
        #     plt.imshow(wordclouds[idx], interpolation='bilinear')
        #     plt.axis("off")
        #     plt.savefig("graphs/wordcloud-(2,2) " + str(i) + "-" + str(idx))
        #     plt.show()
