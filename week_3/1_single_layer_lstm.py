import tensorflow_datasets as tfds
import tensorflow as tf
print(tf.__version__)

dataset, info = tfds.load("imdb_reveiws/subwords8k", with_info=True, as_supervised=True)
train_data, test_data = dataset["train"], dataset["test"]

buffer_size=10000
batch_size=64

train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.padded_batch(batch_size, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_dataset.padded_batch(batch_size, tf.compat.v1.data.get_output_shapes(test_data))


model = tf.keras.Sequential([
	tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(64, activation="relu"),
	tf.keras.layers.Dense(2, activation="sigmoid")
	])

print(model.summary())

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

import matplotlib.pyplot as plt 

def plot_graphs(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history["val_" + string])
	plt.xlabel("Epochs")
	plt.ylabel(string)
	plt.legend([string, "val_" + string])
	plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")