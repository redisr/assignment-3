import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
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
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("Get sentences...")
    # data = get_sentences(X, tokenizer)
    # for i in years:
    # print (i)
    data = get_texts_by_year(X, 2003)

    #TODO: I'll lemmatize the data later
    print("Lemmatize data")
    # print(data)
    # data = lemmatize_data(data)
    # print(data[0])
    # input()
    # vectorize the sentences to ngram (2-grams and 3-grams)
    print("Vectorize data")
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=100)
    tokens = vectorizer.fit_transform(data[:, 1])
    # tokens = vectorizer.fit_transform(data)

    # Run this and go watch some netflix, get some sleep, and check it tomorrow
    wcss = []
    print("K-Means time")
    for clusters in range(3,4):
        print (clusters)
        kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0, n_jobs=-1)
        y_labels = kmeans.fit_predict(tokens)
        silhouette_avg = silhouette_score(tokens, y_labels, sample_size=1000)

        #High silhouette values mean that the clusters are well separated.
        print("For n_clusters =", clusters,
              "The average silhouette_score is :", silhouette_avg)
        wcss.append(kmeans.inertia_)

        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, tokens.shape[0] + (clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tokens, y_labels)

        y_lower = 10
        for i in range(clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[y_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

        plt.show()


    # # plot cost function
    # plt.plot(range(3, 20), wcss)
    # plt.savefig("graphs/cost")
    # plt.show()

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
