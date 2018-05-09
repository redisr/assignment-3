import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from utils import get_texts_by_year, get_years, normalize_data

dataset = pd.read_csv('news_headlines/news_headlines.csv')
X = dataset.iloc[1:, :].values

if __name__ == '__main__':
    years = get_years(X[:, 0])
    for i in years:
        print (i)
        data = get_texts_by_year(X, i)

        data = normalize_data(data)
        # vectorize the sentences to ngram
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
        
        tokens = vectorizer.fit_transform(data[:, 1])
        print ("Number of features " + str(np.shape(tokens)))
        input()
        # Run this and go watch some netflix, get some sleep, and check it tomorrow
        wcss = []
        for clusters in range(3, 30, 2):
            print (clusters)
            kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 400, n_init = 10, n_jobs=7)
            kmeans.fit(tokens)
            wcss.append(kmeans.inertia_)

        # plot cost function
        plt.plot(range(3, 30, 2), wcss)
        plt.savefig("graphs/cost-" + str(i) + "(2, 2)")
        plt.show()
        n_clusters = input()

        # Find clusters
        kmeans = KMeans(n_clusters=int(n_clusters), init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=7)
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
            wordclouds.append(WordCloud().generate(" ".join(data[clusters[idx], 1])))
            plt.imshow(wordclouds[idx], interpolation='bilinear')
            plt.axis("off")
            plt.savefig("graphs/wordcloud-(2,2) " + str(i) + "-" + str(idx))
            plt.show()