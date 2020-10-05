import tensorflow as tf
import numpy as np
import SequenceModel
from tensorflow import keras
from tensorflow.keras import layers


LSTM_DIM = 1028
LSTM_STACKED = 2
ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

LEARNING_RATE = 1e-3
BATCH_SIZE = 200
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.05

CHECKPOINT_PATH = "dist_checkpoints1/model.ckpt"

##################################################################################################
##################################################################################################

#embedding layer that converts nucleotide one hot encodings to corresponding variables
class SequenceEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(SequenceEncoder, self).__init__()
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape, mask=None):
        self.sites_w = self.add_weight(shape=(len(ALPHABET), len(ALPHABET)),
                                        trainable=True,
                                        name="alphabet_embedding",
                                        initializer = tf.constant_initializer(np.eye(len(ALPHABET))))

    #input dim is [batchsize, seq_len, len(ALPHABET)+SequenceModel.LSTM_DIM]
    def call(self, inputs, mask=None):
        one_hot_in = inputs[:,:,:len(ALPHABET)]
        encode_in = tf.linalg.matmul(one_hot_in, self.sites_w)
        return tf.concat((encode_in, inputs[:,:,len(ALPHABET):]), axis=2)


#reduces a sequence of embeddings to a sequence embedding
#and computes distances for subsequent pairs thereafter
class SequenceDecoder(layers.Layer):
    def __init__(self):
        super(SequenceDecoder, self).__init__()

    def call(self, inputs, mask=None):
        # maskf = tf.expand_dims(tf.cast(mask, "float32"), -1)
        # seq_av = tf.reduce_sum(inputs * maskf, axis=1) / tf.reduce_sum(maskf, axis=1)
        # dist = seq_av[::2, :] - seq_av[1::2, :]
        # return tf.norm(dist, ord=1, axis=-1)
        return tf.norm((inputs[::2,:] - inputs[1::2,:]), ord=1, axis=-1)

##################################################################################################
##################################################################################################

def make_model():
    #build the model
    model = tf.keras.Sequential()
    model.add(layers.Masking(mask_value=0.0))
    model.add(SequenceEncoder())
    model.add(layers.LSTM(LSTM_DIM, return_sequences=True))
    model.add(layers.Bidirectional(layers.LSTM(LSTM_DIM, return_sequences=False)))
    # model.add(layers.LSTM(LSTM_DIM, return_sequences=True))
    # model.add(layers.LSTM(LSTM_DIM, return_sequences=True))
    model.add(layers.Dense(LSTM_DIM))
    model.add(SequenceDecoder())

    model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    return model
