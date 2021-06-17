import numpy as np
import re
from sklearn.datasets import load_files

import nltk
# nltk.download('stopwords')

import pickle
from nltk.corpus import stopwords

import os
import random
# import spacy
from spacy.util import minibatch, compounding

# 1 - importing the dataset
movie_data = load_files(r"D:\\FCAI\\4rth year 2nd term\\NLP\\ass2\\txt_sentoken")

# from pos and neg into x(list of 2000 string)
# target into y(numpy array of size 2000 1s and 0s)
X, y = movie_data.data, movie_data.target

# 2 - text processing
# the text may contain numbers special characters and unwanted spaces

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Because of the dataset in bytes format 'b' is appended before every string
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization : reduce the word into dictionary root form
    # cats -> cat
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

# 3 - Converting Text to Numbers
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# max_features set to 1500 because when we convert words to numbers all the unique words are converted to features
# which means that we want to use 1500 most occurring words as features for training our classifier

# min_df set to 5 corresponds to the minimum number of documents that should contain this feature
# include those words that occur in at least 5 documents

# max_df set to 0.7 means that we should include only those words that occur in a maximum of 70% of all the documents
# these words are usually not suitable for classification because they do not provide any unique information about the document

# remove stop words from out text because stop words may not contain any useful information

# The fit_transform function of the CountVectorizer class converts text documents into corresponding numeric features

# Converting into tf-idf
from sklearn.feature_extraction.text import TfidfTransformer

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Training and Testing Sets
# divide data into 20% test set and 80% training set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train the algo
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

# predict the sentiment
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

def test_model(input_data: str = TEST_REVIEW):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )


TEST_REVIEW = """
Transcendently beautiful in moments outside the office, it seems almost
sitcom-like in those scenes. When Toni Colette walks out and ponders
life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide
whether it's slapstick, farce, magical realism, or drama, but the best of it
doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)
"""

train, test = load_training_data(limit=2500)
train_model(train, test)
print("Testing model")
test_model()

# print the accuracy
print(accuracy_score(y_test, y_pred))