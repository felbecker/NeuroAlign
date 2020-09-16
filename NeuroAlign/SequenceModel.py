import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import numpy as np
import MSA


LSTM_DIM = 256
ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X']

LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.05


##################################################################################################
##################################################################################################

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
class SequenceBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return int(np.floor(len(self.split) / BATCH_SIZE)) #steps per epoch

    def __getitem__(self, index):
        batch_indices = np.random.choice(self.split, size=BATCH_SIZE)
        batch = [msa.raw_seq[i] for i in batch_indices]
        batch_lens = [s.shape[0] for s in batch]
        num_steps = max(batch_lens)
        seq = np.zeros((BATCH_SIZE, num_steps, len(ALPHABET)), dtype=np.float32)
        for j,(l,s) in enumerate(zip(batch_lens, batch)):
            seq[j,np.arange(l),s] = 1
        return seq, seq, [None]


train_gen = SequenceBatchGenerator(train)
val_gen = SequenceBatchGenerator(test)

##################################################################################################
##################################################################################################

#embedding layer that converts nucleotide one hot encodings to corresponding variables
class SequenceEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(SequenceEmbedding, self).__init__()

    def build(self, input_shape):
        self.sites_w = self.add_weight(shape=(len(ALPHABET), len(ALPHABET)),
                                        trainable=True,
                                        initializer = tf.constant_initializer(np.eye(len(ALPHABET))))

    #input dim is [batchsize, seq_len, len(ALPHABET)]
    def call(self, inputs):
        return tf.linalg.matmul(inputs, self.sites_w)


#shift such that output i contains forward pass to i-1 and backward pass to i+1 (zero at edges)
#-> output i has never seen position i (which it will predict)
class OutputShift(tf.keras.layers.Layer):
    def __init__(self):
        super(OutputShift, self).__init__()

    #input dim is [batchsize, seq_len, len(ALPHABET)]
    def call(self, inputs):
        #inputs have shape (batch, len, 2*LSTM_DIM)
        #shift such that:
        #out_i = (forward_i-1 , backward_i+1)
        zeros = tf.zeros_like(inputs[:,0:1,:LSTM_DIM])
        forward_shifted = tf.concat([zeros, inputs[:,1:,:LSTM_DIM]], axis=1)
        backward_shifted = tf.concat([inputs[:,:-1,LSTM_DIM:], zeros], axis=1)
        output = tf.concat([forward_shifted, backward_shifted], axis=-1)
        return output

##################################################################################################
##################################################################################################

#build the model
model = tf.keras.Sequential()

#compute the initial embeddings per sequence symbol
model.add(SequenceEmbedding())

#will mask out padded zeros in all downstream layers
model.add(tf.keras.layers.Masking(mask_value=0.0))

#bidirectional LSTMs
model.add(tf.keras.layers.Bidirectional(layer = tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True),
                                        backward_layer = tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True, go_backwards=True),
                                        merge_mode = "concat"))
model.add(tf.keras.layers.Bidirectional(layer = tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True),
                                        backward_layer = tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True, go_backwards=True),
                                        merge_mode = "concat"))

#transform to symbol probabilities, this part will be skipped when shipping the model
model.add(OutputShift())
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(ALPHABET))))
model.add(tf.keras.layers.Activation('softmax'))

##################################################################################################
##################################################################################################

model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                metrics=['categorical_accuracy'])

data = val_gen.__getitem__(0)

model.fit(train_gen,
            validation_data=val_gen,
            epochs = NUM_EPOCHS,
            verbose = 1)
