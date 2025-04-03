import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.tag import CRFTagger

# Generate Data
text = {
    'review' : ['Anton adalah karyawan yang baik. Dia pandai dalam melakukan pekerjaannya. Setiap hari selalu mencuci keris pribadinya sehingga sudah siap ketika akan digunakan.', 'Yudi sangat peduli dengan customer. Dia selalu menanyakan kabar customer sebelum melakukan pekerjaannya.', 'Ardi sangat teliti dalam melakukan pekerjaannya. Hasil pekerjaannya tidak pernah berantakan.'],
    'rating' : [5, 4 ,5],
}

data = pd.DataFrame(data = text)

# Preprocessing
def preprocessing(input):
    case_folded = input.lower()
    tokenize_word = word_tokenize(case_folded)
    return tokenize_word

data['preprosesed'] = data['review'].apply(preprocessing)

# POS Tagging
ct = CRFTagger()
ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
pos_review = ct.tag_sents(data['preprosesed'])

print(pos_review)