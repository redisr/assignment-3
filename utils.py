import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
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

def stem_text(func, text):
    words = text.split(" ")
    stop_words = set(stopwords.words('english'))
    new_words = []
    for word in words:
        # filter stop words and meaningless words len < 3
        if word not in stop_words and len(word) > 2:
            new_words.append(func(word))
    return " ".join(new_words)

def stem_data(texts):
    # alg = WordNetLemmatizer().lemmatize
    function = PorterStemmer().stem
    for i in range(len(texts)):
        texts[i] = stem_text(function, texts[i,1])
    return texts
