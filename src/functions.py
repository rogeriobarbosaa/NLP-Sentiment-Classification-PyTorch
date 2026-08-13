from nltk.corpus import stopwords
import string
from spacy.tokens import Doc

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

def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5*iqr
    upper_limit = q3 + 1.5*iqr

    no_outliers = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

    return no_outliers

# usar com .apply()
def lematiza_tokens(tokens, nlp):
    doc = Doc(nlp.vocab, words=tokens)  # usa tokens, sem re-tokenizar
    for pipe_name in nlp.pipe_names:
        doc = nlp.get_pipe(pipe_name)(doc)
    return [token.lemma_ for token in doc]

# usar com .apply()
def embedding_ft(tokens, ft):
    # o FastText monta o vetor pelos n-gramas, então cobre palavras inéditas; o try protege o caso raro de nenhum n-grama ser conhecido
    vector = []
    for token in tokens:
        try:
            vector.append(ft.wv[token])
        except KeyError:
            continue
    return vector