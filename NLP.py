import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

data = pd.read_csv('tripadvisor_hotel_reviews.csv')
data['review_lowercase'] = data['Review'].str.lower()
en_stopword = stopwords.words('english')

en_stopword.remove('not')

data['review_no_stopword'] = data['review_lowercase'].apply(lambda x: ' '.join(word for word in x.split() if word not in en_stopword))
data['review_no_stopword_no_punct'] = data.apply(lambda x: re.sub(r"[*]", " star", x['review_no_stopword']), axis = 1)
data['review_no_stopword_no_punct'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['review_no_stopword_no_punct']), axis= 1)
data['tokenize'] = data.apply(lambda x: word_tokenize(x['review_no_stopword_no_punct']), axis=1)

ps = PorterStemmer()
data['stemmed'] = data['tokenize'].apply(lambda tokens: [ps.stem(token) for token in tokens])

lemmatize = WordNetLemmatizer()

data['lemmatized'] = data['tokenize'].apply(lambda tokens: [lemmatize.lemmatize(token) for token in tokens])

token_clean = sum(data['lemmatized'], [])

unigram = (pd.Series(nltk.ngrams(token_clean, 1)))
bigram = (pd.Series(nltk.ngrams(token_clean, 2)).value_counts())
print(unigram)
print(bigram)
print(data.info())