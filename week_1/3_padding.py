# code 4

"""
when we feed sequences of number to the neural network, they have to be uniform in size/length
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ["I love my dog", "I love my cat", "you love my dog!", "Do you think my dog is amazing!"]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\n word_index - ", word_index)
print("\n length of word_index - ", len(word_index))

sequences = tokenizer.texts_to_sequences(sentences)
print("\n sequences : ", sequences)

""" PADDING """
# bydefault - pre-padding
padded = pad_sequences(sequences)
print("\n padded sequences : ", padded)

# post padding 
post_padded = pad_sequences(sequences, padding="post")
print("\n post padded sequences : ", post_padded)


# Truncating the sequences 
""" 
we can truncate the sequences by providing maximum length
but if we have truncating, we will loose information
by default it is "pre"
"""
# pre truncation
truncated = pad_sequences(sequences, padding="post", maxlen=5)
print("\n truncated sequences - ", truncated)

# post truncating
post_truncated = pad_sequences(sequences, padding="post", truncating="post", maxlen=5)
print("\n post_truncated : ", post_truncated)

import pdb;pdb.set_trace()
