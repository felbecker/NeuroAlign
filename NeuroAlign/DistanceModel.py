import tensorflow as tf
import numpy as np
import SequenceModel


LSTM_DIM = 1028
LSTM_STACKED = 2
ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.05

CHECKPOINT_PATH = "dist_checkpoints/model.ckpt"

##################################################################################################
##################################################################################################

#embedding layer that converts nucleotide one hot encodings to corresponding variables
class SequenceEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(SequenceEncoder, self).__init__()

    def build(self, input_shape):
        self.sites_w = self.add_weight(shape=(len(ALPHABET), len(ALPHABET)),
                                        trainable=True,
                                        name="alphabet_embedding",
                                        initializer = tf.constant_initializer(np.eye(len(ALPHABET))))

    #input dim is [batchsize, seq_len, len(ALPHABET)+SequenceModel.LSTM_DIM]
    def call(self, inputs):
        one_hot_in = inputs[:,:,:len(ALPHABET)]
        encode_in = tf.linalg.matmul(one_hot_in, self.sites_w)
        return tf.concat((encode_in, inputs[:,:,len(ALPHABET):]), axis=2)


#reduces a sequence of embeddings to a sequence embedding
#and computes distances for subsequent pairs thereafter
class SequenceDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(SequenceDecoder, self).__init__()

    def build(self, input_shape):
        self.a = self.add_weight(shape=(1), trainable=True, name="decoder_weight", initializer = tf.constant_initializer(1))
        self.b = self.add_weight(shape=(1), trainable=True, name="decoder_bias", initializer = tf.constant_initializer(0))

    #input dim is [2*batchsize, seq_len, LSTM_DIM]
    def call(self, inputs):
        reduced_along_seq = tf.math.reduce_mean(inputs, axis=1)
        dist = reduced_along_seq[:,::2] - reduced_along_seq[:,1::2]
        return self.a*tf.norm(dist, keepdims=True) + self.b


##################################################################################################
##################################################################################################

def make_model():
    #build the model
    model = tf.keras.Sequential()
    model.add(SequenceEncoder())
    model.add(tf.keras.layers.Masking(mask_value=0.0))
    model.add(tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True))
    model.add(tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True))
    model.add(SequenceDecoder())

    model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    return model
