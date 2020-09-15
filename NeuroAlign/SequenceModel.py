import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import numpy as np
import MSA


SITE_DIM = 25
LSTM_DIM = 32
ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X']

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 10
VALIDATION_SPLIT = 0.1

#load the data
msa = MSA.Instance("Pfam_very_thin.fasta", ALPHABET, gaps=False)

if not msa.valid:
    print("Invalid data.")
    quit()

np.random.seed(0)

indices = np.array(range(len(msa.raw_seq)))
np.random.shuffle(indices)
train, test = np.split(indices, [int(len(msa.raw_seq)*(1-VALIDATION_SPLIT))])


##################################################################################################
##################################################################################################

#provides batches of protein sequences
class SequenceBatchGenerator(object):
    def __init__(self, split, batch_size):
        self.data = data
        self.split = split
        self.batch_size = batch_size

    def generate(self):
        while True:
            batch = msa.raw_seq[ np.random.randint(self.split, size=self.batch_size) ]
            num_steps = max([s.shape[0] for s in msa.raw_seq[batch]])
            x = np.zeros((self.batch_size, num_steps, len(ALPHABET)))
            y = np.zeros((self.batch_size, num_steps, len(ALPHABET)))
            yield x, y


train_gen = SequenceBatchGenerator(train, BATCH_SIZE)
val_gen = SequenceBatchGenerator(test, BATCH_SIZE)

##################################################################################################
##################################################################################################

#embedding layer that converts nucleotide one hot encodings to corresponding variables
class SequenceEmbedding(keras.layers.Layer):
    def __init__(self):
        super(IntraSNPLayer, self).__init__()

    def build(self, input_shape):
        self.sites_w = self.add_weight(shape=(len(ALPHABET), SITE_DIM), trainable=True)

    #input dim is [batchsize, seq_len, len(ALPHABET)]
    def call(self, inputs):
        return tf.linalg.matmul(inputs, self.sites_w)

##################################################################################################
##################################################################################################

#build the model
model = Sequential()

#compute the initial embeddings per sequence symbol
model.add(SequenceEmbedding())

#will mask out padded zeros in all downstream layers
model.add(tf.keras.layers.Masking(mask_value=0.0))

#bidirectional LSTMs
model.add(tf.keras.layers.Bidirectional(layer = LSTM(LSTM_DIM, return_sequences=True),
                                        backward_layer = LSTM(LSTM_DIM, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(layer = LSTM(LSTM_DIM, return_sequences=True),
                                        backward_layer = LSTM(LSTM_DIM, return_sequences=True)))

#transform to symbol probabilities, this part will be skipped when shipping the model
model.add(tf.keras.layers.TimeDistributed(Dense(len(ALPHABET))))
model.add(tf.keras.layers.Activation('softmax'))

##################################################################################################
##################################################################################################

model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                metrics=['categorical_accuracy'])

model.fit(train_gen.generate(), validation_data=val_gen.generate(),
            batch_size = 1, epochs = NUM_EPOCHS, verbose = 2,
            callbacks=[checkpointer])
