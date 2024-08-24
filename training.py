import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Resolved nltk lookup errors not needed anymore
# nltk.download('all')

# Initialize the WordNetLemmatizer, which will be used for lemmatizing words
lemmatizer = WordNetLemmatizer()

# Load the intents file and parse it as JSON
intents = json.loads(open('intents.json').read())

# Initialize lists to hold processed words, classes (intents), and documents
words = []
classes = []
documents = []

# Define characters to ignore in the text
ignore_letters = ['?', '!', ".", ","]

# Iterate over each intent in the intents file
for intent in intents["intents"]:
    # Iterate over each pattern associated with the current intent
    for pattern in intent["patterns"]:
        # Tokenize the pattern into individual words
        wordlist = nltk.word_tokenize(pattern)
        
        # Add the tokenized words to the words list
        words.extend(wordlist)
        
        # Add the tokenized pattern and its associated tag to the documents list
        documents.append((wordlist, intent['tag']))
        
        # Add the intent tag to the classes list if it's not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    # Create output row for the current intent
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    # Append the bag of words and the output row to training data
    training.append(bag + output_row)

#randomize training data
random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))  


SGD = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = SGD, metrics = ['accuracy'])

hist = model.fit(train_x, train_y, epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot_model.keras', hist)

print('Done')