# imports
import sys
import numpy as np
# accept Vocab.dat and Embedding.dat as command line arguments
vocab = sys.argv[1]
embedding = sys.argv[2]
# load the vocab into a numpy array
vocab_sequence, embedding_sequence = [], []
for word in open(vocab).readlines():
    vocab_sequence.append(word)
vocab_sequence = np.asarray(vocab_sequence)
# load embedding into a numpy array
for embed in open(embedding).readlines():
    embedding_sequence.append(np.asarray(embed))
embedding_sequence = np.asarray(embedding_sequence)
print(embedding_sequence)
print(embedding_sequence.shape)
print(embedding_sequence.size)
