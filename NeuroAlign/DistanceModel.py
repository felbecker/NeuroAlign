import tensorflow as tf
import numpy as np
import SequenceModel
from tensorflow import keras
from tensorflow.keras import layers


##################################################################################################
##################################################################################################

ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

##################################################################################################
##################################################################################################

LSTM_DIM = 1028
ENCODER_DIM = 1000
INPUT_DIM = len(ALPHABET) + 2*SequenceModel.LSTM_DIM
OUTPUT_DIM = 256

LEARNING_RATE = 1e-3
BATCH_SIZE = 300
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.05

CHECKPOINT_PATH = "dist_checkpoints_with_LM/model.ckpt"

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
        self.dense = layers.Dense(ENCODER_DIM, activation="relu")

    def call(self, inputs, mask=None):
        one_hot_in = inputs[:,:,:len(ALPHABET)]
        encode_in = tf.linalg.matmul(one_hot_in, self.sites_w)
        encode_in = tf.concat((encode_in, inputs[:,:,len(ALPHABET):]), axis=2)
        encoded = self.dense(tf.reshape(encode_in, (-1, INPUT_DIM)))
        encoded = tf.reshape(encoded, (tf.shape(encode_in)[0], tf.shape(encode_in)[1], ENCODER_DIM))
        return encoded


#reduces a sequence of embeddings to a sequence embedding
#and computes distances for subsequent pairs thereafter
class ReduceSequences(layers.Layer):
    def __init__(self):
        super(ReduceSequences, self).__init__()

    def call(self, inputs, mask=None):
        maskf = tf.expand_dims(tf.cast(mask, "float32"), -1)
        seq_av = tf.reduce_sum(inputs * maskf, axis=1) / tf.reduce_sum(maskf, axis=1)
        return seq_av


#reduces a sequence of embeddings to a sequence embedding
#and computes distances for subsequent pairs thereafter
class SequenceDecoder(layers.Layer):
    def __init__(self):
        super(SequenceDecoder, self).__init__()
        #self.dense1 = layers.Dense(OUTPUT_DIM)
        #self.relu = layers.Activation("relu")
        #self.dense2 = layers.Dense(1)

    def call(self, inputs):
        #l1 = self.dense1(inputs[::2, :]) + self.dense1(inputs[1::2, :])
        #l1 = self.relu(l1)
        #return self.dense2(l1)
        return tf.norm(inputs[::2, :] - inputs[1::2, :], ord=1, axis=-1)

##################################################################################################
##################################################################################################

def make_model():
    #build the model
    model = tf.keras.Sequential()
    model.add(layers.Masking(mask_value=0.0))
    model.add(SequenceEncoder())
    model.add(layers.LSTM(LSTM_DIM, return_sequences=True))
    model.add(layers.LSTM(LSTM_DIM, return_sequences=True))
    model.add(ReduceSequences())
    model.add(layers.Dense(OUTPUT_DIM, activation="relu"))
    model.add(layers.Dense(OUTPUT_DIM))
    model.add(SequenceDecoder())

    model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return model
