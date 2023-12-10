"""
date: November 2020
multi-class classification
"""

import csv
import tensorflow as tf 
import numpy as np

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# import nltk
from nltk.corpus import stopwords
stopwords_en = list(set(stopwords.words("english")))


#hyperparamters
vocab_size = 1000
embedding_size = 16
max_length = 120
trunc_type = "post"
padding_type = "post"
oov_token = "<OOV>"
training_portion = 0.8
num_epochs=30


sentences = []
labels = []

with open("datasets/bbc-text.csv", "r") as csvfile:
	reader = csv.reader(csvfile, delimiter=",")
	next(reader)

	for row in reader:
		labels.append(row[0])
		sentence = row[1]

		for word in stopwords_en:
			token = " " + word + " "
			sentence = sentence.replace(token, " ")
		sentences.append(sentence)

import pdb;pdb.set_trace()

print(len(sentences))
print(len(labels))

train_size = int(len(sentences)* training_portion)
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]


print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))


### tokenization 
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_sequence = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_sequence = np.array(label_tokenizer.texts_to_sequences(validation_labels))

### build model
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=max_length),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(24, activation="relu"),
	tf.keras.layers.Dense(6, activation="softmax")
	])


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("\n model summary : ", model.summary())

history = model.fit(train_padded, training_label_sequence, epochs=num_epochs, validation_data=(validation_padded, validation_label_sequence), verbose=2)

import pdb;pdb.set_trace()