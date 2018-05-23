# imports
import sys
import numpy as np
# accept Vocab.dat and Embedding.dat as command line arguments
vocab = sys.argv[1]
embedding = sys.argv[2]
# load the vocab into a numpy array
vocab_sequence, embedding_sequence = [], []
vocab_sequence = np.loadtxt(vocab, dtype=str)
# load embedding into a numpy array
embedding_sequence = np.loadtxt(embedding)
for dimension in range(16):
    lowest_10 = np.argsort(embedding_sequence[:, dimension])[-10:]
    highest_10 = np.argsort(embedding_sequence[:, dimension])[:10]
    print("Dimension: ", dimension + 1)
    lowest_10 = np.take(vocab_sequence, lowest_10)
    highest_10 = np.take(vocab_sequence, highest_10)
    lowest_10 = np.insert(lowest_10, 0, "LOWEST 10")
    highest_10 = np.insert(highest_10, 0, "HIGHEST 10")
    # print(np.hstack((lowest_10, highest_10)))
    np.savetxt(sys.stdout, np.vstack((lowest_10, highest_10)), fmt='%s')
