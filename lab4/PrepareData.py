# Imports
from keras.preprocessing.text import Tokenizer
import re
import operator
import numpy as np
np.set_printoptions(suppress=True)
import sys
# Defining files
Stoker = sys.argv[1]
Austen = sys.argv[2]
with open(Austen, encoding="ISO-8859-1") as Austen_open:
    Austen_read = Austen_open.read()
    # Austen_read=Austen_read.encode('utf-8','ignore').decode('utf-8','ignore')
    # removing punctutation except that required to end sentences
    p = re.compile('[^\w\.;:\s]')
    # filter for the Mr. Mrs. etc
    q = re.compile('(?<=Mr|rs)\.')
    # removing Chapter #
    r = re.compile('Chapter\s\d+')
    # problem: handle periods within paragraphs
    # Austen_read=q.sub('',Austen_read)
    Austen_read = p.sub('', Austen_read)
    Austen_read = q.sub('', Austen_read)
    Austen_read = r.sub('', Austen_read)
    p = re.compile('\n')
    Austen_read = p.sub(' ', Austen_read)
    Austen_read = re.split(pattern='[.;:?\n]', string=Austen_read)
    Austen_read = list(map(lambda x: x.lstrip().rstrip(), Austen_read))
    Austen_read = list(map(lambda x: x.lstrip().rstrip(), Austen_read))
    for line_number, line in enumerate(Austen_read):
        if (len(line) in range(0, 10)):
            del (Austen_read[line_number])
    # print(len(Austen_read))
    # print(*Austen_read,sep='\n')
with open(Stoker) as Stoker_open:
    Stoker_read = Stoker_open.read()
    #fix time
    p = re.compile('P\.\s?M\.', re.I)
    Stoker_read = p.sub('', Stoker_read)
    p = re.compile('[0-9]')
    Stoker_read = p.sub('', Stoker_read)
    #remove personal comments
    p = re.compile('\(.+\)', re.I)
    Stoker_read = p.sub('', Stoker_read)
    #remove diary entry starts of the form _......--
    p = re.compile('_.+--', re.I)
    Stoker_read = p.sub('', Stoker_read)
    #since -- seems to seperate sentences
    p = re.compile('--')
    Stoker_read = p.sub('.', Stoker_read)
    #filter for the Mr. Mrs. etc
    p = re.compile('(?<=Mr|rs)\.', re.I)
    Stoker_read = p.sub('', Stoker_read)
    #remove ******** lines
    p = re.compile('.+\*+.+', re.I)
    Stoker_read = p.sub('', Stoker_read)
    #remove quotation and other random marks
    p = re.compile('[{}()\-_~`#$%^&*+=\\\|:\"\']')
    Stoker_read = p.sub('', Stoker_read)
    #print(Stoker_read)
    #try splitting using ending punctuation.
    #remove newlines
    p = re.compile('\n')
    Stoker_read = p.sub(' ', Stoker_read)
    p = re.compile('[A-Z]{2,}')
    Stoker_read = p.sub(' ', Stoker_read)
    Stoker_read = re.split(pattern='[.;?!]', string=Stoker_read)
    Stoker_read = list(map(lambda x: x.lstrip().rstrip(), Stoker_read))
    for line_number, line in enumerate(Stoker_read):
        if (len(line) in range(0, 10)):
            del (Stoker_read[line_number])
    # print(len(Stoker_read))
    # print(*Stoker_read,sep='\n')
samples = Austen_read + Stoker_read
# print(samples)
tokenizer = Tokenizer(num_words=12000)
tokenizer.fit_on_texts(samples)
Austen_sequences = np.asarray(tokenizer.texts_to_sequences(Austen_read))
Stoker_sequences = np.asarray(tokenizer.texts_to_sequences(Stoker_read))
np.random.shuffle(Austen_sequences)
np.random.shuffle(Stoker_sequences)
# print(Austen_sequences)
# print(Stoker_sequences)
Austen_split = int(Austen_sequences.shape[0] * 0.75)
Stoker_split = int(Stoker_sequences.shape[0] * 0.75)
Austen_file = Austen[:-4]
Stoker_file = Stoker[:-4]
# for line in range(0,Stoker_split):
# np.savetxt(Stoker[:-4],Stoker_sequences[line],fmt='%i')
# np.savetxt(sys.stdout,Stoker_sequences[0:Stoker_split],fmt='%s',delimiter=',')
np.savetxt(
    Stoker_file + '.train',
    Stoker_sequences[0:Stoker_split],
    fmt='%s',
    delimiter=',')
np.savetxt(
    Stoker_file + '.test',
    Stoker_sequences[Stoker_split:],
    fmt='%s',
    delimiter=',')
np.savetxt(
    Austen_file + '.train',
    Austen_sequences[0:Austen_split],
    fmt='%s',
    delimiter=',')
np.savetxt(
    Austen_file + '.test',
    Austen_sequences[Austen_split:],
    fmt='%s',
    delimiter=',')
word_index = tokenizer.word_index
word_index['FeelsBadMan'] = 0
sorted_word_index = np.asarray(
    sorted(word_index.items(), key=operator.itemgetter(1)))
np.savetxt('Vocab.dat', sorted_word_index[:12000, 0], fmt="%s")
