# code 2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


sentences = ["I love my dog", "I love my cat", "you love my dog!", "Do you think my dog is amazing!"]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)


"""
for unseen words, it will not assign any number, it will simply skip them, e.g. really, loves, manatee.
"""

test_data = ["I really love my dog", "my dog loves my manatee"]
test_sequences = tokenizer.texts_to_sequences(test_data)
print("\n test_sequences : ", test_sequences)


"""
we need lot of training data to create big vocabulary , otherwise we could end up with sentences like test data
other way is to put some value when we encounter any unseen word. There is a property of tokenizer which will help us to do that

"""

