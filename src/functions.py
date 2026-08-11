from nltk.corpus import stopwords
import string

# usar com .apply()
def remove_stopwords_punctuation(text):
    # remove stopwords
    stops = stopwords.words("portuguese")
    no_stopwords = [p for p in text if p not in stops]

    # remove pontuação
    puncts = string.punctuation
    no_punctuation = [p for p in no_stopwords if p not in puncts]

    return no_punctuation

def token_outlier(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)

    iqr = q3 - q1

    upper_limit = column - 1.5*iqr
    lower_limit = column + 1.5*iqr

    outliers = column < lower_limit | column > upper_limit

    return outliers