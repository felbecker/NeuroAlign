import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os

jobid = 777#int(os.getenv('SLURM_ARRAY_TASK_ID'))

##################################################################################################
##################################################################################################

ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']


##################################################################################################
##################################################################################################

#number of message passing iterations to perform
NUM_ITERATIONS = 5

#the dimensions for the different internal representations
SITE_DIM = 50
SEQ_LSTM_DIM = 1000
CONS_LSTM_DIM = 300

#hidden layer sizes for the MLPs
ENCODER_LAYERS = [200]
SEQUENCE_MSGR_LAYERS = [300, 300]
CONSENSUS_MSGR_LAYERS = [300, 300]

#if false, each message passing iteration uses unique parameters
SHARED_ITERATIONS = True#True if jobid-1 < 4 else False

#if true, total loss is the sum of losses from all iterations
LOSS_OVER_ALL_ITERATIONS = True

# f true, seconary loss terms for gap lengths between all pairs of sites and
# for column aminoacid distributions decoded from consensus are used for training
USE_MULTI_TASK_LOSSES = False


#percentage of families not used for training
VALIDATION_SPLIT = 0.05#0.01


#maximum number of sites in a batch
#must be at least as large as the sum of the two longest sequences in all families

BATCH_SIZE = 1200

#number of splits for the membership updates
#this step is typically the memory-bottleneck of the model
#by choosing a value > 1, runtime can be traded for reduced
#memory requirements during inference
#for instance, inference on a computer with 8GB RAM for
#1000 input sequences of average length 150 is possible with COL_BATCHES = 20
#this parameter has no effect during training
COL_BATCHES = 1

LEARNING_RATE = 4e-5

NUM_EPOCHS = 40

NAME = "alignment_model"
CHECKPOINT_PATH = NAME+"/model.ckpt"

SEQ_IN_DIM = len(ALPHABET)+2

__POS_WEIGHT = [40,90,40,90]
POS_WEIGHT = 80#__POS_WEIGHT[(jobid-1)%4]#70#155.5
NEG_WEIGHT = 0.1#0.5

__LAST_ITERATION_WEIGHT = [1,1,8,8]
LAST_ITERATION_WEIGHT = 1#__LAST_ITERATION_WEIGHT[(jobid-1)%4]


MEM_LOSS_WEIGHT = 1
GAP_LOSS_WEIGHT = 0.1
AA_DIST_LOSS_WEIGHT = 1

COLUMN_OVEREXTENSION = 1

##################################################################################################
##################################################################################################

CFG_TXT = ("NeuroAlign config  \n" +
      " iterations: "+ str(NUM_ITERATIONS)+
      " - site dim: "+ str(SITE_DIM)+
      " - seq lstm dim: "+ str(SEQ_LSTM_DIM)+
      " - cons lstm dim: "+ str(CONS_LSTM_DIM)+
      " - encoder: "+ str(ENCODER_LAYERS)+ " \n"+
      " - sequence messenger: "+ str(SEQUENCE_MSGR_LAYERS)+
      " - consensus messenger: "+ str(CONSENSUS_MSGR_LAYERS)+" \n"+
      " - shared iterations: "+ str(SHARED_ITERATIONS)+
      " - loss over all iterations: "+ str(LOSS_OVER_ALL_ITERATIONS)+" \n"+
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

#decodes all combinations of sites and consensus to a soft alignment matrix
class MembershipDecoder(layers.Layer):
    def __init__(self):
        super(MembershipDecoder, self).__init__()

    def build(self, input_shape):
        self.in_sites = layers.Dense(32, activation="relu")
        self.in_cons = layers.Dense(32, activation="relu")
        self.decoder = MLP([32])
        self.out_transf = layers.Dense(1)

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 6:
            raise Exception('MembershipDecoder must be called on a list of exactly 6 tensors ')

        sites = self.in_sites(inputs[0])
        cons = self.in_cons(inputs[1])

        sites = tf.expand_dims(sites, 1)
        cons = tf.expand_dims(cons, 0)

        comb = tf.reshape(sites + cons, (-1, 32))

        logits = self.out_transf(self.decoder(comb))
        logits = tf.reshape(logits, (tf.shape(inputs[0])[0], tf.shape(inputs[1])[0]))

        M = tf.nn.softmax(logits, axis=-1)

        M_c_lower = tf.cumsum(M, axis=-1, reverse=True)
        M_c_lower = tf.scatter_nd(tf.expand_dims(inputs[2], -1), M_c_lower, shape=(inputs[4] * inputs[5], tf.shape(inputs[1])[0]))
        M_c_lower = tf.reshape(M_c_lower, (inputs[4], inputs[5], tf.shape(inputs[1])[0]))
        M_lower = tf.math.cumprod(tf.nn.relu(1 - M_c_lower), axis=1, exclusive=True)

        M_c_higher = tf.cumsum(M, axis=-1)
        M_c_higher = tf.scatter_nd(tf.expand_dims(inputs[3], -1), M_c_higher, shape=(inputs[4] * inputs[5], tf.shape(inputs[1])[0]))
        M_c_higher = tf.reshape(M_c_higher, (inputs[4], inputs[5], tf.shape(inputs[1])[0]))
        M_higher = tf.math.cumprod(tf.nn.relu(1 - M_c_higher), axis=1, exclusive=True)

        M_lower = tf.gather(tf.reshape(M_lower, (-1, tf.shape(inputs[1])[0]) ), inputs[2], axis=0)
        M_higher = tf.gather(tf.reshape(M_higher, (-1, tf.shape(inputs[1])[0]) ), inputs[3], axis=0)

        return M * M_lower * M_higher

##################################################################################################
##################################################################################################

#decodes gap lengths for all combinations of sites
#by soft-argmaxing their column membership vectors
class GapDecoder(layers.Layer):
    def __init__(self):
        super(GapDecoder, self).__init__()

    def build(self, input_shape):
        self.in_sites = layers.Dense(32, activation="relu")
        self.out_transf = layers.Dense(1)

    def call(self, gathered_sequences):
        seq = self.in_sites(gathered_sequences)
        gaps = tf.expand_dims(seq, 1) + tf.expand_dims(seq, 0)
        gaps = tf.reshape(gaps, (-1, 32))
        gaps = self.out_transf(gaps)
        gaps = tf.reshape(gaps, (tf.shape(gathered_sequences)[0], tf.shape(gathered_sequences)[0]))
        return gaps

##################################################################################################
##################################################################################################

#decodes amino acid distributions from columns (consensus sequence)
class AminoDistDecoder(layers.Layer):
    def __init__(self):
        super(AminoDistDecoder, self).__init__()

    def build(self, input_shape):
        self.in_cons = layers.Dense(32, activation="relu")
        self.out_transf = layers.Dense(len(ALPHABET))

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 2:
            raise Exception('AminoDistDecoder must be called on a list of exactly 2 tensors ')
        aa_dist = self.out_transf(self.in_cons(inputs[0]))
        aa_dist = tf.nn.softmax(aa_dist, axis=-1) * tf.expand_dims(tf.reduce_sum(inputs[1], axis=0), -1)
        return aa_dist

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
################################4/5wENU41iG-fcZ19h08AXYxWtk4vAywDxkcqvQdH_ucPW8ODHX42dicA##################################################################

def make_model(training = True, output_representations = False):
    #define inputs
    sequences = keras.Input(shape=(None,SEQ_IN_DIM), name="sequences")
    sequence_gather_indices = keras.Input(shape=(), name="sequence_gather_indices", dtype=tf.int32)
    rev_sequence_gather_indices = keras.Input(shape=(), name="rev_sequence_gather_indices", dtype=tf.int32)
    initial_memberships = keras.Input(shape=(None,), name="initial_memberships")

    encoder = MLP(ENCODER_LAYERS+[SITE_DIM])
    mem_decoder = MembershipDecoder()
    if USE_MULTI_TASK_LOSSES:
        gap_decoder = GapDecoder()
        aa_dist_decoder = AminoDistDecoder()

    #networks
    if SHARED_ITERATIONS:
        seq_lstm = [layers.Bidirectional(layers.LSTM(SEQ_LSTM_DIM, return_sequences=True),
                                        backward_layer=layers.LSTM(SEQ_LSTM_DIM, return_sequences=True, go_backwards=True))]*NUM_ITERATIONS
        seq_dense = [layers.TimeDistributed(layers.Dense(SITE_DIM))]*NUM_ITERATIONS
        cons_lstm = [layers.Bidirectional(layers.LSTM(CONS_LSTM_DIM, return_sequences=True),
                                        backward_layer=layers.LSTM(CONS_LSTM_DIM, return_sequences=True, go_backwards=True))]*NUM_ITERATIONS
        cons_dense = [layers.TimeDistributed(layers.Dense(2*SITE_DIM))]*NUM_ITERATIONS
        seq_to_cons_messenger = [Messenger(SEQUENCE_MSGR_LAYERS)]*NUM_ITERATIONS
        cons_to_seq_messenger = [Messenger(CONSENSUS_MSGR_LAYERS)]*NUM_ITERATIONS
    else:
        seq_lstm = [layers.Bidirectional(layers.LSTM(SEQ_LSTM_DIM, return_sequences=True),
                                        backward_layer=layers.LSTM(SEQ_LSTM_DIM, return_sequences=True, go_backwards=True)) for _ in range(NUM_ITERATIONS)]
        seq_dense = [layers.TimeDistributed(layers.Dense(SITE_DIM)) for _ in range(NUM_ITERATIONS)]
        cons_lstm = [layers.Bidirectional(layers.LSTM(CONS_LSTM_DIM, return_sequences=True),
                                        backward_layer=layers.LSTM(CONS_LSTM_DIM, return_sequences=True, go_backwards=True)) for _ in range(NUM_ITERATIONS)]
        cons_dense = [layers.TimeDistributed(layers.Dense(2*SITE_DIM)) for _ in range(NUM_ITERATIONS)]
        seq_to_cons_messenger = [Messenger(SEQUENCE_MSGR_LAYERS) for _ in range(NUM_ITERATIONS)]
        cons_to_seq_messenger = [Messenger(CONSENSUS_MSGR_LAYERS) for _ in range(NUM_ITERATIONS)]

    #encode the sequences
    masked_sequences = layers.Masking(mask_value=0.0)(sequences)
    encoded_sequences = layers.TimeDistributed(encoder)(masked_sequences)
    gathered_initial_sequences = tf.gather(tf.reshape(encoded_sequences, (-1, SITE_DIM) ), sequence_gather_indices, axis=0)
    gathered_sequences = gathered_initial_sequences

    #initial consensus
    #av_site = tf.reduce_mean(gathered_sequences, axis=0, keepdims=True)
    #av_site = tf.concat([av_site, av_site], axis=-1)
    #consensus = tf.tile(av_site, [tf.shape(initial_memberships)[1], 1])
    #consensus = tf.zeros((tf.shape(initial_memberships)[1], 2*SITE_DIM))
    consensus = layers.Dense(2*SITE_DIM)(tf.expand_dims(tf.range(tf.shape(initial_memberships)[1], dtype=tf.float32) / tf.cast(tf.shape(initial_memberships)[1], dtype=tf.float32), 1))

    gathered_sequences_Concat = layers.Concatenate()([gathered_initial_sequences, gathered_sequences])
    M = mem_decoder([gathered_sequences_Concat, consensus, sequence_gather_indices, rev_sequence_gather_indices, tf.shape(sequences)[0], tf.shape(sequences)[1]])
    #consensus = tf.matmul(M, gathered_sequences_Concat, transpose_a=True)

    outputs = []

    for i in range(NUM_ITERATIONS):

        messages_cons_to_seq = cons_to_seq_messenger[i]([consensus, M])
        messages_cons_to_seq = tf.scatter_nd(tf.expand_dims(sequence_gather_indices, -1), messages_cons_to_seq,
                                                shape=(tf.shape(sequences)[0] * tf.shape(sequences)[1], CONSENSUS_MSGR_LAYERS[-1]))
        messages_cons_to_seq = tf.reshape(messages_cons_to_seq, (tf.shape(sequences)[0], tf.shape(sequences)[1], CONSENSUS_MSGR_LAYERS[-1]))
        concat_sequences = layers.Concatenate()([masked_sequences, encoded_sequences, messages_cons_to_seq])
        encoded_sequences = seq_dense[i](seq_lstm[i](concat_sequences)) #will always mask
        gathered_sequences = tf.gather(tf.reshape(encoded_sequences, (-1, SITE_DIM) ), sequence_gather_indices, axis=0)

        gathered_sequences_Concat = layers.Concatenate()([gathered_initial_sequences, gathered_sequences])

        messages_seq_to_cons = seq_to_cons_messenger[i]([gathered_sequences_Concat, tf.transpose(M)])
        concat_cons = layers.Concatenate()([consensus, messages_seq_to_cons])
        consensus = cons_dense[i](cons_lstm[i](tf.expand_dims(concat_cons, 0)))
        consensus = tf.squeeze(consensus, axis=0)
        #consensus = cons_network[i](concat_cons)

        M = mem_decoder([gathered_sequences_Concat, consensus, sequence_gather_indices, rev_sequence_gather_indices, tf.shape(sequences)[0], tf.shape(sequences)[1]])

        if training:
            if LOSS_OVER_ALL_ITERATIONS:
                    outputs.append(layers.Lambda(lambda x: x, name="mem"+str(i))(tf.linalg.matmul(M, M, transpose_b=True)))
                    if USE_MULTI_TASK_LOSSES:
                        outputs.append(layers.Lambda(lambda x: x, name="gap"+str(i))(gap_decoder(gathered_sequences_Concat)))
                        outputs.append(layers.Lambda(lambda x: x, name="aa_dist"+str(i))(aa_dist_decoder([consensus, M])))
            elif i == (NUM_ITERATIONS-1):
                    outputs.append(layers.Lambda(lambda x: x, name="mem")(tf.linalg.matmul(M, M, transpose_b=True)))
                    if USE_MULTI_TASK_LOSSES:
                        outputs.append(layers.Lambda(lambda x: x, name="gap")(gap_decoder(gathered_sequences_Concat)))
                        outputs.append(layers.Lambda(lambda x: x, name="aa_dist")(aa_dist_decoder([consensus, M])))
        else:
            outputs.append(M)
            if output_representations:
                outputs.extend([gathered_sequences_Concat, consensus])

    model = keras.Model(
        inputs=[sequences, sequence_gather_indices, rev_sequence_gather_indices, initial_memberships],
        outputs=outputs
    )

    return model
