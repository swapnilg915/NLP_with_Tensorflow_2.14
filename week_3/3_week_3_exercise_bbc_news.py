"""
BBC news dataset
to download dataset = wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv -O ./training_cleaned.csv
date: 4th NOV 2020	
"""

import os
import json
import csv
import random
import tensorflow as tf 
import numpy as np 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

embedding_dim = 100
max_length = 16
trunc_type="post"
padding_type="post"
oov_token = "<OOV>"
training_size=160000
test_portion = 0.1

corpus=[]
num_sentences = 0

with open("training_cleaned.csv") as fs:
	reader=csv.reader(fs, delimeter=",")
	for row in reader:
		list_item=[]
		list_item.append(row[5])
		this_label=row[0]
		if this_label == "0": list_item.append(0)
		else: list_item.append(1)
		num_sentences += 1
		corpus.append(list_item)


sentences=[]
labels=[]

random.shuffle(corpus)
for idx in range(len(corpus)):
	sentences.append(corpus[idx][0])
	labels.append(corpus[idx][1])

tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_seqeunce(sequences)
