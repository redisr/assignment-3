import numpy as np

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
