import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os

jobid = 9999#int(os.getenv('SLURM_ARRAY_TASK_ID'))

##################################################################################################
##################################################################################################

ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

##################################################################################################
##################################################################################################

#number of message passing iterations to perform
NUM_ITERATIONS = 2

#the dimensions for the different internal representations
SITE_DIM = 50
SEQ_LSTM_DIM = 512
CONS_LSTM_DIM = 512

#hidden layer sizes for the MLPs
ENCODER_LAYERS = [256, 256]
SEQUENCE_MSGR_LAYERS = [256,256]
CONSENSUS_MSGR_LAYERS = [256,256]

#if false, each message passing iteration uses unique parameters
SHARED_ITERATIONS = False#True if jobid-1 < 4 else False


VALIDATION_SPLIT = 0.25#0.01


#maximum number of sites in a batch
#must be at least as large as the sum of the two longest sequences in all families
BATCH_SIZE = 2000

#number of splits for the membership updates
#this step is typically the memory-bottleneck of the model
#by choosing a value > 1, runtime can be traded to reduce
#memory requirements during inference
COL_BATCHES = 1

LEARNING_RATE = 1e-5
NUM_EPOCHS = 200

NAME = "alignment_model"+str(jobid-1)

SEQ_IN_DIM = len(ALPHABET)

__POS_WEIGHT = [40,90,40,90]
POS_WEIGHT = 7#__POS_WEIGHT[(jobid-1)%4]#70#155.5
NEG_WEIGHT = 1#0.5

__LAST_ITERATION_WEIGHT = [1,1,8,8]
LAST_ITERATION_WEIGHT = 3#__LAST_ITERATION_WEIGHT[(jobid-1)%4]

##################################################################################################
##################################################################################################

CHECKPOINT_PATH = NAME+"/model.ckpt"

##################################################################################################
##################################################################################################

CFG_TXT = ("NeuroAlign config  \n" +
      " iterations: "+ str(NUM_ITERATIONS)+
      " - site dim: "+ str(SITE_DIM)+
      " - seq lstm dim: "+ str(SEQ_LSTM_DIM)+
      " - cons lstm dim: "+ str(CONS_LSTM_DIM)+
      " - encoder: "+ str(ENCODER_LAYERS)+ " \n"+
      " - sequence messenger: "+ str(SEQUENCE_MSGR_LAYERS)+
      " - consensus messenger: "+ str(CONSENSUS_MSGR_LAYERS)+
      " - shared iterations: "+ str(SHARED_ITERATIONS)+ " \n"+
      " - pos weight: "+ str(POS_WEIGHT)+
      " - neg weight: "+ str(NEG_WEIGHT)+
      " - batch: "+ str(BATCH_SIZE)+
      " - last iteration weight: "+ str(LAST_ITERATION_WEIGHT)+
      " - learning rate: "+ str(LEARNING_RATE))

print(CFG_TXT, flush=True)

##################################################################################################
##################################################################################################

#MLP with relu activations and layer normalized output
#the last dense layer is unactivated
class MLP(layers.Layer):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes

    def build(self, input_shape):
        self.layers = [layers.Dense(l, activation="relu") for l in self.layer_sizes[:-1]]
        self.layers.append(layers.Dense(self.layer_sizes[-1]))
        self.layers.append(layers.LayerNormalization())

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        cur = inputs
        for l in self.layers:
            cur = l(cur)
        return cur

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.layer_sizes[-1])

##################################################################################################
##################################################################################################

#decodes all combinations of site (resp. consensus) and column representations
#to soft memberships
class MembershipDecoder(layers.Layer):
    def __init__(self):
        super(MembershipDecoder, self).__init__()

    def build(self, input_shape):
        self.softmax = layers.Activation("softmax")

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 2:
            raise Exception('MembershipDecoder must be called on a list of exactly 2 tensors ')
        sites = tf.expand_dims(inputs[0], 1)
        consensus = tf.expand_dims(inputs[1], 0)
        norm = tf.norm(sites - consensus, ord=1, axis=-1)
        return self.softmax(-norm)

##################################################################################################
##################################################################################################

#computes messages based on inputs according to given soft memberships
class Messenger(layers.Layer):
    def __init__(self, layers):
        super(Messenger, self).__init__()
        self.layers = layers

    def build(self, input_shape):
        self.messenger = MLP(self.layers)

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 2:
            raise Exception('Messenger must be called on a list of exactly 2 tensors ')
        messages = self.messenger(inputs[0])
        messages_to_cons = tf.linalg.matmul(inputs[1], messages)
        return messages_to_cons

##################################################################################################
##################################################################################################

def make_model():
    #define inputs
    sequences = keras.Input(shape=(None,SEQ_IN_DIM), name="sequences")
    sequence_gatherer = keras.Input(shape=(None,), name="sequence_gatherer")
    initial_memberships = keras.Input(shape=(None,), name="initial_memberships")

    encoder = MLP(ENCODER_LAYERS+[SITE_DIM])
    mem_decoder = MembershipDecoder()
    #networks
    if SHARED_ITERATIONS:
        seq_lstm = [layers.LSTM(SEQ_LSTM_DIM, return_sequences=True)]*NUM_ITERATIONS
        seq_dense = [layers.TimeDistributed(layers.Dense(SITE_DIM))]*NUM_ITERATIONS
        cons_lstm = [layers.LSTM(CONS_LSTM_DIM, return_sequences=True)]*NUM_ITERATIONS
        cons_dense = [layers.TimeDistributed(layers.Dense(SITE_DIM))]*NUM_ITERATIONS
        seq_to_cons_messenger = [Messenger(SEQUENCE_MSGR_LAYERS)]*NUM_ITERATIONS
        cons_to_seq_messenger = [Messenger(CONSENSUS_MSGR_LAYERS)]*NUM_ITERATIONS
    else:
        seq_lstm = [layers.LSTM(SEQ_LSTM_DIM, return_sequences=True) for _ in range(NUM_ITERATIONS)]
        seq_dense = [layers.TimeDistributed(layers.Dense(SITE_DIM)) for _ in range(NUM_ITERATIONS)]
        cons_lstm = [layers.LSTM(CONS_LSTM_DIM, return_sequences=True) for _ in range(NUM_ITERATIONS)]
        cons_dense = [layers.TimeDistributed(layers.Dense(SITE_DIM)) for _ in range(NUM_ITERATIONS)]
        seq_to_cons_messenger = [Messenger(SEQUENCE_MSGR_LAYERS) for _ in range(NUM_ITERATIONS)]
        cons_to_seq_messenger = [Messenger(CONSENSUS_MSGR_LAYERS) for _ in range(NUM_ITERATIONS)]

    #encode the sequences
    masked_sequences = layers.Masking(mask_value=0.0)(sequences)
    encoded_sequences = layers.TimeDistributed(encoder)(masked_sequences)
    gathered_initial_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(encoded_sequences, (-1, SITE_DIM) ))
    gathered_sequences = gathered_initial_sequences

    #initial consensus
    consensus = tf.zeros((tf.shape(initial_memberships)[1], SITE_DIM))

    M = mem_decoder([gathered_sequences, consensus])

    mem_sq_out = []

    for i in range(NUM_ITERATIONS):

        gathered_sequences_Concat = layers.Concatenate()([gathered_initial_sequences, gathered_sequences])
        messages_seq_to_cons = seq_to_cons_messenger[i]([gathered_sequences_Concat, tf.transpose(M)])
        concat_cons = layers.Concatenate()([consensus, messages_seq_to_cons])
        consensus = cons_dense[i](cons_lstm[i](tf.expand_dims(concat_cons, axis=0)))
        consensus = tf.squeeze(consensus, axis=0)

        messages_cons_to_seq = cons_to_seq_messenger[i]([consensus, M])
        messages_cons_to_seq = tf.linalg.matmul(sequence_gatherer, messages_cons_to_seq, transpose_a = True)
        messages_cons_to_seq = tf.reshape(messages_cons_to_seq, (tf.shape(sequences)[0], tf.shape(sequences)[1], CONSENSUS_MSGR_LAYERS[-1]))
        concat_sequences = layers.Concatenate()([masked_sequences, encoded_sequences, messages_cons_to_seq])
        encoded_sequences = seq_dense[i](seq_lstm[i](concat_sequences)) #will always mask
        gathered_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(encoded_sequences, (-1, SITE_DIM) ))

        M = mem_decoder([gathered_sequences, consensus])
        M_squared = tf.linalg.matmul(M, M, transpose_b=True)
        mem_sq_out.append(layers.Lambda(lambda x: x, name="mem"+str(i))(M_squared))

    model = keras.Model(
        inputs=[sequences, sequence_gatherer, initial_memberships],
        outputs=mem_sq_out
    )

    return model
