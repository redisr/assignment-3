import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from utils import get_texts_by_year, get_years

dataset = pd.read_csv('news_headlines/news_headlines.csv')
X = dataset.iloc[1:, :].values

# TODO split data by year
years = get_years(X[:, 0])
for i in years:
    print (i)
    data = get_texts_by_year(X, i)

    # vectorize the sentences to ngram
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    tokens = vectorizer.fit_transform(data[:, 1])
    
    # Run this and go watch some netflix, get some sleep, and check it tomorrow
    wcss = []
    for clusters in range(2,20):
        print (clusters)
        kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0, n_jobs=3)
        kmeans.fit(tokens)
        wcss.append(kmeans.inertia_)

    # plot cost function
    plt.plot(range(2, 20), wcss)
    plt.savefig("graphs/cost-" + str(i) + "(1, 2)")
    plt.show()
    n_clusters = input()

    # Find clusters
    kmeans = KMeans(n_clusters=int(n_clusters), init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=3)
    kmeans.fit(tokens)
    y_kmeans = kmeans.fit_predict(tokens)

    sorted_indexes = np.argsort(y_kmeans)
    texts = dict()

    # separate clusters
    clusters = dict()
    for indexes in sorted_indexes:
        if not(y_kmeans[indexes] in clusters):
            print (y_kmeans[indexes])
            clusters[y_kmeans[indexes]] = []
        clusters[y_kmeans[indexes]].append(indexes)


    # make word clouds out of each cluster
    wordclouds = []
    for idx in clusters:
        wordclouds.append(WordCloud().generate(" ".join(X[clusters[idx], 1])))
        plt.imshow(wordclouds[idx], interpolation='bilinear')
        plt.axis("off")
        plt.savefig("graphs/wordcloud-(1,2) " + str(i) + "-" + str(idx))
        plt.show()