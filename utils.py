import numpy as np
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from nltk.corpus import stopwords

def get_year(date):
    return str(date)[0:4]

def get_years(dates):
    years = np.array(dates / 10000, dtype=np.int)
    return np.unique(years)


def get_texts_by_year(texts, year):
    text_idx = []
    year = str(year)
    for i in range(len(texts)):
        if year == get_year(texts[i, 0]):
            text_idx.append(i)
    
    print (np.array(text_idx))
    return texts[np.array(text_idx)]

def normalize_text(stem_lem, text):
    words = text.split(" ")
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            word = stem_lem(word)
            #word = wlen.lemmatize(word)
            new_words.append(word)
    return " ".join(new_words)

def normalize_data(texts):
    wlem = WordNetLemmatizer()
    stemmer = LancasterStemmer()
    for i in range(len(texts)):
        texts[i, 1] = normalize_text(wlem.lemmatize, texts[i, 1])
    return texts
