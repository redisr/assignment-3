import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

dataset = pd.read_csv('news_headlines/news_headlines.csv')
X = dataset.iloc[1:, :].values

# TODO split data by year

# vectorize the sentences to ngram
vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 3))
tokens = vectorizer.fit_transform(X[:, 1])

# Run this and go watch some netflix, get some sleep, and check it tomorrow
wcss = []
for i in range(2,28):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0, n_jobs=-1)
    kmeans.fit(tokens)
    wcss.append(kmeans.inertia_)


# Find clusters
kmeans = KMeans(n_clusters=21, init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=-1)
kmeans.fit(tokens)
y_kmeans = kmeans.fit_predict(tokens)

sorted_indexes = np.argsort(y_kmeans)
texts = dict()

# separate clusters
cluster = dict()
for i in sorted_indexes:
    if not(y_kmeans[i] in cluster):
        print (y_kmeans[i])
        cluster[y_kmeans[i]] = []
    cluster[y_kmeans[i]].append(i)


# make word clouds out of each cluster
wordclouds = []
for i in cluster:
    wordclouds.append(WordCloud().generate(" ".join(X[cluster[i], 1])))

plt.imshow(wordclouds[0], interpolation='bilinear')
plt.axis("off")
plt.show()