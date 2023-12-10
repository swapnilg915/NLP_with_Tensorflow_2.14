"""
use tensorflow 2.x.
eager execution is default in tf 2
install tensorflow-datasets
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb["train"], imdb["test"]

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for sent, lab in train_data:
	training_sentences.append(str(sent.numpy().decode('utf8')))
	training_labels.append(str(lab.numpy()))

for sent, lab in test_data:
	testing_sentences.append(str(sent.numpy().decode('utf8')))
	testing_labels.append(str(lab.numpy()))

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

import pdb;pdb.set_trace()

# ### tokenization
# vocab_size = 10000
# embedding_dim = 16
# max_length = 120
# trunc_type = "post"
# oov_tok = "<OOV>"
# num_epochs=10

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(training_sentences)
# word_index = tokenizer.word_index
# sequences = tokenizer.texts_to_sequences(training_sentences)
# padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

# testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
# testing_padded = pad_sequences(testing_sequences, maxlen=max_length)


# ### neural network
# model = tf.keras.Sequential([
# 	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
# 	tf.keras.layers.Flatten(), 
# 	tf.keras.layers.Dense(6, activation="relu"),
# 	tf.keras.layers.Dense(1, activation="sigmoid")
# 	]
# 	)	

# """
# flatten is equal to "globalaveragepooling1d"
# """


# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# print("\n model summary : ", model.summary())


# model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))





vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))