from keras import models, layers, optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import sys
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
stoker = sys.argv[1]
austen = sys.argv[2]
MAXLEN = 114
num_words = 12000
batch_size = 128
stoker_sequence, austen_sequence = [], []
# Using eval because I can't think of anything else at this hour
for line in open(stoker).readlines():
    stoker_sequence.append(np.asarray(eval(line)))
for line in open(austen).readlines():
    austen_sequence.append(np.asarray(eval(line)))
stoker_sequence = np.asarray(stoker_sequence)
austen_sequence = np.asarray(austen_sequence)
# checking the sequences from the saved files
# np.savetxt(sys.stdout,stoker_sequence,fmt='%s')
austen_len = austen_sequence.shape[0]
stoker_len = stoker_sequence.shape[0]
train_austen = sequence.pad_sequences(austen_sequence, maxlen=MAXLEN)
train_stoker = sequence.pad_sequences(stoker_sequence, maxlen=MAXLEN)


def generator(stoker, austen, stoker_len, austen_len, batch_size=128):
    stoker_index, austen_index = 0, 0
    while True:
        batch_features = np.zeros((batch_size, MAXLEN))
        batch_labels = np.zeros((batch_size, 1))
        for i in range(batch_size):
            # choose a book for number of batch_size.0:stoker1:austen
            book_choice = np.random.choice([0, 1])
            if (book_choice == 0):
                # stoker
                batch_features[i] = stoker[stoker_index]
                stoker_index = (stoker_index + 1) % stoker_len
            else:
                batch_features[i] = austen[austen_index]
                austen_index = (austen_index + 1) % austen_len
            batch_labels[i] = book_choice
        yield batch_features, batch_labels


# time to create the generators.
# train/validation split is 90% to 10%
stoker_split = int(stoker_len * 0.90)
austen_split = int(austen_len * 0.90)
stoker_split2 = int(stoker_len * 0.10)
austen_split2 = int(austen_len * 0.10)
# for the train, I can use the first few without haveing to slice the data
# but for the last one, I need to freaking slice the numpy array
train_gen = generator(
    train_stoker,
    train_austen,
    stoker_split,
    austen_split,
    batch_size=batch_size)
val_gen = generator(
    train_stoker[-stoker_split2:],
    train_austen[-austen_split2:],
    stoker_split2,
    austen_split2,
    batch_size=batch_size)
model = models.Sequential()
model.add(layers.Embedding(12000, 16, input_length=MAXLEN))
model.add(layers.LSTM(32))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# checkpoint
checkpointer = ModelCheckpoint(
    filepath='PorV.h5',
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max')
model.summary()
history = model.fit_generator(
    train_gen,
    steps_per_epoch=(max(stoker_split, austen_split) // batch_size) + 1,
    epochs=20,
    validation_data=val_gen,
    validation_steps=(max(stoker_split2, austen_split2) // batch_size) + 1,
    callbacks=[checkpointer],
    verbose=1)
hst = history.history
x_axis = range(0, len(hst['acc']))
plt.plot(x_axis, hst['acc'], 'bo')
plt.plot(x_axis, hst['val_acc'], 'ro')
plt.show()
plt.savefig('try1.png')
# model.save('PorV.h5')
# next steps: Embedding.datasets
"""model = load_model('PorV.h5')
model2 = models.Sequential()
model2.add(
    layers.Embedding(
        10000, 16, input_length=MAXLEN, weights=model.layers[0].get_weights()))
# build numpy array to predict
temp = []
for i in range(10000):
    temp.append([i])
temp = sequence.pad_sequences(temp, maxlen=MAXLEN)
activations = model2.predict(temp)
intermediate = activations[:, 0, :]
np.savetxt('Embedding.dat', intermediate, fmt='%2e')"""
