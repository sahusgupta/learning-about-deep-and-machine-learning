from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from flask import Flask, request
from json import dumps
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
nltk.download('stopwords')
nltk.download('wordnet')

def process(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

data = pd.read_csv('data/search.csv')
vectorizer = TfidfVectorizer()


vectorizer.fit_transform(data['search'])
data['search'].apply(process)
X_train, X_test, y_train, y_test = train_test_split(data['search'], data['appropriate'], test_size=0.7, random_state=42)
model = make_pipeline(TfidfVectorizer(), ComplementNB())
model.fit(X_train, y_train)
preds = model.predict(X_test)
fname = 'model.sav'
pickle.dump(model, open(fname, 'wb'))
def appropriate(text):
    labels = ['inappropriate', 'appropriate']
    return labels[model.predict([text])[0]]
