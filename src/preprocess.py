# src/preprocess.py
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

def preprocess_text(text):
    stemmer = PorterStemmer()
    spam_important_words = {'you','now','free','win','winner','claim','prize','cash'}
    stopwords_set = set(stopwords.words('english')) - spam_important_words

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    return text

def preprocess_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df