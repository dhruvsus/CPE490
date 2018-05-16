from keras import models, layers, optimizers
import sys
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
stoker=sys.argv[1]
austen=sys.argv[2]
MAXLEN=114
stoker_sequence, austen_sequence, features, labels=[],[],[],[]
# Using eval because I can't think of anything else at this hour
for line in open(stoker).readlines():
    stoker_sequence.append(np.asarray(eval(line)))
stoker_sequence=np.asarray(stoker_sequence)
for line in open(austen).readlines():
    austen_sequence.append(np.asarray(eval(line)))
austen_sequence=np.asarray(austen_sequence)
austen_len=austen_sequence.shape[0]
stoker_len=stoker_sequence.shape[0]
train_austen=list(sequence.pad_sequences(austen_sequence,maxlen=MAXLEN))
train_stoker=list(sequence.pad_sequences(stoker_sequence,maxlen=MAXLEN))
#print(train_stoker)
#print(train_austen)
"""def generator(austen, stoker, batch_size=128):
    while 1:
        for i in range(0,batch_size):
            if(i%2==0):
                if(train_stoker):
                    features.append(train_stoker.pop())
                else:
                    train_stoker=sequence.pad_sequences(stoker_sequence,maxlen=MAXLEN)
                    features.append(train_stoker.pop())
                labels.append(i%2)
            else:
                if(train_austen):
                    features.append(train_austen.pop())
                else:
                    train_austen=sequence.pad_sequences(austen_sequence,maxlen=MAXLEN)
                    features.append(train_austen.pop())
                labels.append(i%2)
        yield features,labels
train_gen=generator(austen,stoker,batch_size=128)"""
model = models.Sequential()
model.add(layers.Embedding(12000, 32))
model.add(layers.LSTM(18))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#history = model.fit(train_gen,steps_per_epoch=50,epochs=5, verbose=1)
