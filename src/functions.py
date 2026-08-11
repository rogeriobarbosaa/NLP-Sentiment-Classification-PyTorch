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

def token_outlier(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5*iqr
    upper_limit = q3 + 1.5*iqr

    outliers = (df[column] < lower_limit) | (df[column] > upper_limit)

    return outliers