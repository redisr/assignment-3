import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from utils import get_texts_by_year, get_years, lemmatize_data

# http://ai.intelligentonlinetools.com/ml/word-embeddinigs-machine-learning/
# https://groups.google.com/forum/#!searchin/gensim/kmeans%7Csort:date/gensim/xLdkU2mbUwo/v9N4gtY0BwAJ
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb
# TODO: Try tu use pre-trained word2vec models and try to infer vectors from the words in the input.
def getFeaturesW2V(data):
    #apply word2vec to extract features from the sentences
    tokens = []

    for i in range(data.shape[0]):
        aux_tokens = data[i].split(" ")
        tokens.append(aux_tokens)

    print("Running Word2Vec")
    model = Word2Vec(tokens, min_count=3)
    print("Word2Vec is ready to go")
    word2vec_dict = {}
    words = model.wv.vocab.keys()  # order from model.wv.syn0

    for i in words:
        word2vec_dict[i] = model[i]

    tokens = np.array([word2vec_dict[i].T for i in words])
    return tokens


dataset = pd.read_csv('news_headlines/news_headlines.csv')
X = dataset.iloc[1:, :].values

# data = lemmatize_data(data)
# vectorize the sentences to ngram
# vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=10000)
# tokens = vectorizer.fit_transform(data[:, 1])

# try to use word2vec to create features
tokens = getFeaturesW2V(X[:,1])

# Run this and go watch some netflix, get some sleep, and check it tomorrow
wcss = []
for clusters in range(3,20):
    print (clusters)
    kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(tokens)
    wcss.append(kmeans.inertia_)

# plot cost function
plt.plot(range(3, 20), wcss)
#plt.savefig("graphs/cost-")
plt.show()
n_clusters = input("número de clusters: ")

# Find clusters
kmeans = KMeans(n_clusters=int(n_clusters), init='k-means++', max_iter=10000, n_init=10, random_state=0)
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
    # plt.savefig("graphs/wordcloud-(2,2) " + str(i) + "-" + str(idx))
    plt.show()
