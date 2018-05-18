from keras import models, layers, optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import sys
from keras import backend as K
from matplotlib import pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
MAXLEN = 114
model_file = sys.argv[1]
model = load_model(model_file)
test_sequence = []
for line in sys.stdin.readlines():
    test_sequence.append(np.asarray(eval(line)))
test_sequence = np.asarray(test_sequence)
test_sequence = sequence.pad_sequences(test_sequence, maxlen=MAXLEN)
test_label = model.predict_classes(test_sequence)
print(test_label.sum()/test_label.size)
# np.savetxt(sys.stdout, test_label, fmt='%d')
