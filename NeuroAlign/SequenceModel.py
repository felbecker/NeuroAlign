import tensorflow as tf
import numpy as np


LSTM_DIM = 512
LSTM_STACKED = 2
ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

LEARNING_RATE = 1e-3
BATCH_SIZE = 800
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.05

CHECKPOINT_PATH = "seq_checkpoints2/model.ckpt"

##################################################################################################
##################################################################################################

#embedding layer that converts nucleotide one hot encodings to corresponding variables
class SequenceEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(SequenceEmbedding, self).__init__()

    def build(self, input_shape):
        self.sites_w = self.add_weight(shape=(len(ALPHABET), len(ALPHABET)),
                                        trainable=True,
                                        name="alphabet_embedding",
                                        initializer = tf.constant_initializer(np.eye(len(ALPHABET))))

    #input dim is [batchsize, seq_len, len(ALPHABET)]
    def call(self, inputs):
        return tf.linalg.matmul(inputs, self.sites_w)


#shift such that output i contains forward pass to i-1 and backward pass to i+1 (zero at edges)
#-> output i has never seen position i (which it will predict)
class OutputShift(tf.keras.layers.Layer):
    def __init__(self):
        super(OutputShift, self).__init__()

    def compute_mask(self, inputs, mask=None):
        return mask

    #input dim is [batchsize, seq_len, len(ALPHABET)]
    def call(self, inputs, mask=None):
        #tf.print(inputs, summarize=-1)
        #inputs have shape (batch, len, 2*LSTM_DIM)
        #shift such that:
        #out_i = (forward_i-1 , backward_i+1)
        zeros = tf.zeros_like(inputs[:,0:1,:LSTM_DIM])
        forward_shifted = tf.concat([zeros, inputs[:,:-1,:LSTM_DIM]], axis=1)
        backward_shifted = tf.concat([inputs[:,1:,LSTM_DIM:], zeros], axis=1)
        output = tf.concat([forward_shifted, backward_shifted], axis=-1)
        return output


class StackedLSTM(tf.keras.layers.Layer):
    def __init__(self,
                    go_backwards=False,
                    return_sequences=True,
                    return_state=False,
                    name="StackedLSTM",
                    trainable=True,
                    dtype=tf.float32):
        super(StackedLSTM, self).__init__(name=name, trainable=trainable, dtype=dtype)
        self.lstms = []
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        for i in range(LSTM_STACKED):
            self.lstms.append(tf.keras.layers.LSTM(LSTM_DIM,
                                                    go_backwards=go_backwards,
                                                    return_sequences=return_sequences,
                                                    return_state=return_state,
                                                    trainable=trainable,
                                                    dtype=dtype))

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })
        return config

    def call(self, inputs, mask=None):
        cur = inputs
        for lstm in self.lstms:
            cur = lstm(cur, mask=mask)
        return cur

##################################################################################################
##################################################################################################

def make_model():
    #build the model
    model = tf.keras.Sequential()

    #compute the initial embeddings per sequence symbol
    model.add(SequenceEmbedding())

    #will mask out padded zeros in all downstream layers
    model.add(tf.keras.layers.Masking(mask_value=0.0))

    #bidirectional LSTMs
    model.add(tf.keras.layers.Bidirectional(layer = StackedLSTM(),
                                            merge_mode = "concat"))
    model.add(OutputShift())

    #transform to symbol probabilities, this part will be skipped when shipping the model
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(ALPHABET))))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    metrics=['categorical_accuracy'])

    model.build(input_shape=(BATCH_SIZE,None,len(ALPHABET)))

    return model
