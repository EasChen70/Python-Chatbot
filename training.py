import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('all')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wordlist = nltk.word_tokenize(pattern)
        words.append(wordlist)
        documents.append((wordlist, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

