import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

NUM_ITERATIONS = 3
SITE_DIM = 64
COL_DIM = 256
ENCODER_LAYERS = [256,256]
COL_MSGR_LAYERS = [256,256]
SEQ_MSGR_LAYERS = [256,256]
DECODER_LAYERS = [100]

VALIDATION_SPLIT = 0.01

#maximum number of sites in a batch
#must be at least as large as the sum of the two longest sequences in all families
BATCH_SIZE = 5000
LEARNING_RATE = 2e-4
MEM_LOSS = 1
RP_LOSS = 0#1
GAP_LOSS = 0#1
COL_LOSS = 0#1
NUM_EPOCHS = 500

CHECKPOINT_PATH = "alignment_checkpoints2/model.ckpt"


print("iterations: ", NUM_ITERATIONS,
      " site_dim: ", SITE_DIM,
      " col_dim: ", COL_DIM,
      " encoder: ", ENCODER_LAYERS,
      "col_msg: ", COL_MSGR_LAYERS,
      "seq_msg: ", SEQ_MSGR_LAYERS,
      "dec: ", DECODER_LAYERS,
      "mem_loss: ", MEM_LOSS,
      "rp_loss: ", RP_LOSS,
      "gap_loss: ", GAP_LOSS,
      "col_loss: ", COL_LOSS, flush=True)

##################################################################################################
##################################################################################################

#MLP with relu activations and layer normalized output
#the last layer is unactivated
class MLP(layers.Layer):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes

    def build(self, input_shape):
        self.layers = [layers.Dense(l, activation="relu") for l in self.layer_sizes[:-1]]
        self.layers.append(layers.LayerNormalization())
        self.layers.append(layers.Dense(self.layer_sizes[-1]))

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

#decodes all representations for all sites and columns
#to a soft membership representation
class MembershipDecoder(layers.Layer):
    def __init__(self):
        super(MembershipDecoder, self).__init__()

    def build(self, input_shape):
        self.decoder_s = layers.Dense(DECODER_LAYERS[0])
        self.decoder_c = layers.Dense(DECODER_LAYERS[0])
        self.decoder_relu = layers.Activation("relu")
        self.decoder_sm = layers.Activation("softmax")
        self.decoder = MLP(DECODER_LAYERS)
        self.out_trans = layers.Dense(1)
        self.logit_w = self.add_weight(shape=(1),
                                        trainable=True,
                                        name="logit_w",
                                        initializer = tf.constant_initializer(1))

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 3:
            raise Exception('MembershipDecoder must be called on a list of exactly 3 tensors ')
        seq_dec = self.decoder_s(inputs[0])
        col_dec = self.decoder_c(inputs[1])
        sequence_lengths = inputs[2]
        logits = tf.expand_dims(seq_dec, 1) + tf.expand_dims(col_dec, 0) #use broadcasting to efficiently get all combinations
        logits = self.decoder_relu(tf.reshape(logits, (-1, DECODER_LAYERS[0])))
        logits = self.out_trans(self.decoder(logits))
        logits = tf.reshape(logits, (-1, tf.shape(inputs[1])[0]))
        M_c = self.decoder_sm(logits)
        M_s = M_c
        # exp_logits = tf.exp(logits)
        # exp_logits = tf.reshape(exp_logits, (-1, tf.shape(inputs[1])[0]))
        # M_c = exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
        # segment_ids = tf.repeat(tf.range(tf.shape(sequence_lengths)[0], dtype=tf.int32), sequence_lengths)
        # segment_sum = tf.repeat(tf.math.segment_sum(exp_logits, segment_ids), sequence_lengths, axis=0)
        # M_s = exp_logits*self.logit_w / (segment_sum+1)
        return M_c, M_s

#decodes all secondary ouputs
class SecDecoder(layers.Layer):
    def __init__(self):
        super(SecDecoder, self).__init__()

    def build(self, input_shape):
        self.col_decoder = layers.Dense(len(ALPHABET)+1, activation="softmax")

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 3:
            raise Exception('SecDecoder must be called on a list of exactly 3 tensors ')
        M = inputs[0]
        columns = inputs[1]
        sequence_lengths = inputs[2]
        ncol_f = tf.cast(tf.shape(M)[1], dtype=tf.float32)
        colrange = tf.reshape(tf.range(ncol_f, dtype=tf.float32), (-1,1))
        soft_argmax = tf.matmul(M, colrange)
        gaps = soft_argmax[1:,:] - soft_argmax[:-1,:] - 1
        indices= tf.cast(tf.reshape(tf.math.cumsum(sequence_lengths), (-1,1)), dtype=tf.int64)
        values = tf.ones((tf.shape(sequence_lengths)[0]-1), dtype=tf.bool)
        st = tf.sparse.SparseTensor(indices[:-1]-1, values, [tf.cast(tf.reduce_sum(sequence_lengths)-1, dtype=tf.int64)])
        remove_seq_ends = tf.math.logical_not(tf.sparse.to_dense(st))
        gaps_in = tf.boolean_mask(gaps, remove_seq_ends) / ncol_f
        gaps_start = tf.gather_nd(soft_argmax, tf.concat([tf.zeros((1,1), dtype=tf.int64), indices[:-1]], axis=0)) / ncol_f
        gaps_end = (ncol_f-tf.gather_nd(soft_argmax, indices-1)-1) / ncol_f
        relative_positions = soft_argmax/ncol_f
        col_dist = self.col_decoder(columns)
        return relative_positions, gaps_start, gaps_in, gaps_end, col_dist

##################################################################################################
##################################################################################################

def make_model():
    #define inputs
    seq_in_dim = len(ALPHABET)+1
    sequences = keras.Input(shape=(None,seq_in_dim), name="sequences")
    sequence_lengths = keras.Input(shape=(), name="sequence_lengths", dtype=tf.int32)
    sequence_gatherer = keras.Input(shape=(None,), name="sequence_gatherer")
    column_priors_c = keras.Input(shape=(None,), name="column_priors_c")
    column_priors_s = keras.Input(shape=(None,), name="column_priors_s")

    #networks
    seq_lstm = layers.LSTM(SITE_DIM, return_sequences=True)
    col_lstm = layers.LSTM(COL_DIM, return_sequences=True)
    encoder = MLP(ENCODER_LAYERS+[SITE_DIM])
    col_messenger = MLP(COL_MSGR_LAYERS)
    seq_messenger = MLP(SEQ_MSGR_LAYERS)
    mem_decoder = MembershipDecoder()
    sec_decoder = SecDecoder()

    def message_seq_to_col(gathered_sequences, M):
        messages_from_seqs = seq_messenger(gathered_sequences)
        messages_to_cols = tf.matmul(M, messages_from_seqs, transpose_a = True)
        return messages_to_cols

    def message_col_to_seq(columns, M):
        messages_from_cols = col_messenger(columns)
        messages_to_seqs = tf.linalg.matmul(M, messages_from_cols)
        messages_to_seqs = tf.linalg.matmul(sequence_gatherer, messages_to_seqs, transpose_a = True)
        return tf.reshape(messages_to_seqs, (tf.shape(sequences)[0], tf.shape(sequences)[1], COL_MSGR_LAYERS[-1]))

    #encode the sequences
    masked_sequences = layers.Masking(mask_value=0.0)(sequences)
    encoded_sequences = layers.TimeDistributed(encoder)(masked_sequences)
    gathered_initial_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(sequences, (-1, seq_in_dim) ))

    #initial memberships
    M_c = column_priors_c
    M_s = column_priors_s

    #initial columns
    columns = tf.ones((tf.shape(M_c)[1], COL_DIM))

    for _ in range(NUM_ITERATIONS):

        # Concatenate applys keras.All to the masks of all concatenated inputs
        #that means if one of them (initial_sequences) has a masked value, all of them will for the duration of the loop
        concat_sequences = layers.Concatenate()([sequences, encoded_sequences, message_col_to_seq(columns, M_c)])
        encoded_sequences = seq_lstm(concat_sequences) #will always mask

        gathered_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(encoded_sequences, (-1, SITE_DIM) ))
        seq_concat = layers.Concatenate()([gathered_initial_sequences, gathered_sequences])

        concat_columns = layers.Concatenate()([columns, message_seq_to_col(seq_concat, M_c)])
        columns = tf.squeeze(col_lstm(tf.expand_dims(concat_columns, axis=0)), axis=0)

        M_c, M_s = mem_decoder([seq_concat, columns, sequence_lengths])

    #M = M_c + M_s - M_s*M_c
    relative_positions, gaps_start, gaps_in, gaps_end, col_dist = sec_decoder([M_c, columns, sequence_lengths])
    M_squared = M_c#tf.linalg.matmul(M_c, M_c, transpose_b=True)

    #name outputs by passing to identity lambdas..
    M_squared = layers.Lambda(lambda x: x, name="mem")(M_squared)
    relative_positions = layers.Lambda(lambda x: x, name="rp")(relative_positions)
    gaps_start = layers.Lambda(lambda x: x, name="gs")(gaps_start)
    gaps_in = layers.Lambda(lambda x: x, name="g")(gaps_in)
    gaps_end = layers.Lambda(lambda x: x, name="ge")(gaps_end)
    col_dist = layers.Lambda(lambda x: x, name="col")(col_dist)

    model = keras.Model(
        inputs=[sequences, sequence_gatherer, column_priors_c, column_priors_s, sequence_lengths],
        #outputs=[M_squared, relative_positions, gaps_start, gaps_in, gaps_end, col_dist],
        outputs=[M_squared],
    )

    model.compile(loss={"mem" : "categorical_crossentropy"},
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss_weights={"mem" : MEM_LOSS},
                    metrics={"mem" : [keras.metrics.Precision(), keras.metrics.Recall()]})

    # model.compile(loss={"mem" : "categorical_crossentropy",
    #                     "rp" : "mse",
    #                     "gs" : "mse",
    #                     "g" : "mse",
    #                     "ge" : "mse",
    #                     "col" : "categorical_crossentropy"},
    #                 optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    #                 loss_weights={"mem" : MEM_LOSS,
    #                                 "rp" : RP_LOSS,
    #                                 "gs" : GAP_LOSS,
    #                                 "g" : GAP_LOSS,
    #                                 "ge" : GAP_LOSS,
    #                                 "col" : COL_LOSS},
    #                 metrics={"mem" : [keras.metrics.Precision(), keras.metrics.Recall()]})

    return model
