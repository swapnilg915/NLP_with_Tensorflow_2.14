# code 1

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ["I love my dog", "I love my cat", "you love my dog"]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\n word_index - ", word_index)
print("\n length of word_index - ", len(word_index))


"""
punctuations are removed, including spaces
it lower cases the words
"""

# tokenizer with OOV

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\n word_index - ", word_index)
print("\n length of word_index - ", len(word_index))

test_data = ["I really love my dog", "my dog loves my manatee"]
test_sequences = tokenizer.texts_to_sequences(test_data)
print("\n test_sequences : ", test_sequences)



