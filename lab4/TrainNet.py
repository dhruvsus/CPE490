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
stoker_sequence, austen_sequence = [], []
# Using eval because I can't think of anything else at this hour
for line in open(stoker).readlines():
    stoker_sequence.append(np.asarray(eval(line)))
for line in open(austen).readlines():
    austen_sequence.append(np.asarray(eval(line)))
stoker_sequence = np.asarray(stoker_sequence)
austen_sequence = np.asarray(austen_sequence)

austen_len = austen_sequence.shape[0]
stoker_len = stoker_sequence.shape[0]
train_austen = list(sequence.pad_sequences(austen_sequence, maxlen=MAXLEN))
train_stoker = list(sequence.pad_sequences(stoker_sequence, maxlen=MAXLEN))


def generator(stoker, austen, batch_size=128):
    orig_stoker_train, temp_stoker = list(stoker), list(stoker)
    orig_austen_train, temp_austen = list(austen), list(austen)
    batch_features = np.zeros((batch_size, MAXLEN))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            book=np.random.choice([0,1])
            if book == 0:
                if temp_stoker is None:
                    temp_stoker = list(orig_stoker_train)
                    batch_features[i] = temp_stoker.pop()
                else:
                    batch_features[i] = temp_stoker.pop()
            else:
                if temp_austen is None:
                    temp_austen = list(orig_austen_train)
                    batch_features[i] = temp_austen.pop()
                else:
                    batch_features[i] = temp_austen.pop()
            batch_labels[i] = book
            yield batch_features, batch_labels
batch_size = 128
# time to create the generators.
# train/validation split is 90% to 10%
austen_split=int(austen_len*0.99)
stoker_split=int(stoker_len*0.99)
train_gen = generator(
    train_stoker[:stoker_split],
    train_austen[:austen_split],
    batch_size=batch_size)
val_gen = generator(
    train_stoker[stoker_split:],
    train_austen[austen_split:],
    batch_size=batch_size)
model = models.Sequential()
model.add(layers.Embedding(10000, 16, input_length=MAXLEN))
model.add(layers.LSTM(32))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit_generator(
    train_gen,
    steps_per_epoch=int(max(stoker_len * 0.9, austen_len * 0.9) // batch_size),
    epochs=2,
    validation_data=val_gen,
    validation_steps=int(
        max(stoker_len * 0.1, austen_len * 0.1) // batch_size),
        verbose=1)
hst = history.history
x_axis = range(0, len(hst['acc']))
plt.plot(x_axis, hst['acc'], 'bo')
plt.plot(x_axis, hst['val_acc'], 'ro')
plt.savefig('try1.png')
model.save('PorV.h5')
# next steps: Embedding.datasets
model = load_model('PorV.h5')
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
np.savetxt('Embedding.dat', intermediate, fmt='%2e')
