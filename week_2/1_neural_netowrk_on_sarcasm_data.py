"""
date: November 2020
binary classification
"""

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# hyper parameters
vocab_size= 1000
oov_token="<OOV>"
max_length=16
embedding_dim=16
trunc_type="post"
padding="post"
training_size=20000
num_epochs = 10

# read data
with open("/home/swapnil/Coursera/NLP_with_Tensorflow/week_1/datasets/sarcasm_dataset/Sarcasm_Headlines_Dataset_v2.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for dic in datastore:
	sentences.append(dic["headline"])
	labels.append(dic["is_sarcastic"])

# split data in train and validation
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = np.array(labels[:training_size])
testing_labels = np.array(labels[training_size:])

tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences=tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, truncating=trunc_type, padding=padding, maxlen=max_length)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, truncating=trunc_type, padding=padding, maxlen=max_length)


# build model
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(6, activation="relu"),
	tf.keras.layers.Dense(1,activation="sigmoid")
	])


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("\n summary : ", model.summary())

model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)