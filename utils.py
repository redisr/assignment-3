import numpy as np
from nltk.stem import WordNetLemmatizer

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

def lemmatize_text(wlen, text):
    words = text.split(" ")
    to_remove = []
    for i in range(len(words)):
        # words with less than 3 letters are discarted
        if len(words[i]) < 3:
            to_remove.append(i)
        else:
            words[i] = wlen.lemmatize(words[i])
    words = np.delete(words, to_remove)
    return " ".join(words)

def lemmatize_data(texts):
    wlem = WordNetLemmatizer()
    for i in range(len(texts)):
        texts[i] = lemmatize_text(wlem, texts[i])
    return texts
