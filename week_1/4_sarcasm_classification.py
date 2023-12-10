""" sarcasm in news headlines dataset by Rishabh Misra

is_sarcastic = 0/1
headline
article_link


we are only going to deal with headlines

samples = 26709
labels = 2
"""

# read data
import json

with open("/home/swapnil/Coursera/NLP_with_Tensorflow/week_1/datasets/sarcasm_dataset/Sarcasm_Headlines_Dataset_v2.json", 'r') as f:
    datastore = json.load(f)
# with open("/home/swapnil/Coursera/NLP_with_Tensorflow/week_1/datasets/sarcasm_dataset/Sarcasm_Headlines_Dataset_v2.json", "r", encoding="utf-8") as fs:
# 	datastore = json.load(fs)
# datastore = json.load(open("/home/swapnil/Coursera/NLP_with_Tensorflow/week_1/datasets/sarcasm_dataset/Sarcasm_Headlines_Dataset_v2.json"))


sentences = []
labels = []
urls = []

for dic in datastore:
	sentences.append(dic["headline"])
	labels.append(dic["is_sarcastic"])
	urls.append(dic["article_link"])


import pdb;pdb.set_trace()

# tokenization and padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\n vocabulary size : ", len(word_index))
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post", maxlen=40)

print("\n padded shape : ", padded.shape)
print("\n padded 1st example - ", padded[0])

