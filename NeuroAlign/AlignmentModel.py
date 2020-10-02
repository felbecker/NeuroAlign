import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

NUM_ITERATIONS = 1
SITE_DIM = 128
COL_DIM = 512
ENCODER_LAYERS = [256, 128]
COL_MSGR_LAYERS = [256,256]
SEQ_MSGR_LAYERS = [256,256]
DECODER_LAYERS = [128]

VALIDATION_SPLIT = 0.01

#maximum number of sites in a batch
#must be at least as large as the sum of the two longest sequences in all families
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
MEM_LOSS = 1
RP_LOSS = 0.5
GAP_LOSS = 0.5
COL_LOSS = 0.8
NUM_EPOCHS = 100

CHECKPOINT_PATH = "alignment_checkpoints/model.ckpt"

##################################################################################################
##################################################################################################

#layer normalized MLP with relu activations and layer normalized output
class MLP(layers.Layer):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes

    def build(self, input_shape):
        self.layers = [layers.Dense(l, activation="relu") for l in self.layer_sizes[:-1]]
        self.layers.append(layers.Dense(self.layer_sizes[-1]))
        self.layer_norm = layers.LayerNormalization()

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        cur = inputs
        for l in self.layers:
            cur = l(cur)
        return self.layer_norm(cur)

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
        self.decoder_s = layers.Dense(DECODER_LAYERS[0], activation="relu")
        self.decoder_c = layers.Dense(DECODER_LAYERS[0], activation="relu")
        self.decoder = MLP(DECODER_LAYERS)
        self.out_trans = layers.Dense(1)

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 3:
            raise Exception('Decode must be called on a list of exactly 3 tensors ')
        seq_dec = self.decoder_s(inputs[0])
        col_dec = self.decoder_c(inputs[1])
        sequence_lengths = inputs[2]
        logits = tf.expand_dims(seq_dec, 1) + tf.expand_dims(col_dec, 0) #use broadcasting to efficiently get all combinations
        _logits = tf.reshape(logits, (-1, DECODER_LAYERS[0]))
        _logits = self.out_trans(self.decoder(_logits))
        exp_logits = tf.exp(_logits)
        exp_logits = tf.reshape(exp_logits, (-1, tf.shape(inputs[1])[0]))
        M_c = exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
        segment_ids = tf.repeat(tf.range(tf.shape(sequence_lengths)[0], dtype=tf.int32), sequence_lengths)
        segment_sum = tf.repeat(tf.math.segment_sum(exp_logits, segment_ids), sequence_lengths, axis=0)
        M_s = exp_logits / segment_sum
        M = M_c + M_s - M_c*M_s
        return M

#decodes all secondary ouputs
class SecDecoder(layers.Layer):
    def __init__(self):
        super(SecDecoder, self).__init__()

    def build(self, input_shape):
        self.col_decoder = layers.Dense(len(ALPHABET)+1, activation="softmax")

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 3:
            raise Exception('Decode must be called on a list of exactly 2 tensors ')
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
    sequences = keras.Input(shape=(None,len(ALPHABET)+2), name="sequences")
    sequence_lengths = keras.Input(shape=(), name="sequence_lengths", dtype=tf.int32)
    sequence_gatherer = keras.Input(shape=(None,), name="sequence_gatherer")
    column_priors = keras.Input(shape=(None,), name="column_priors")

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
    initial_sequences = layers.TimeDistributed(encoder)(masked_sequences)
    gathered_initial_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(initial_sequences, (-1, SITE_DIM) ))
    encoded_sequences = initial_sequences

    #initial memberships
    M = column_priors

    #initial columns
    columns = tf.ones((tf.shape(M)[1], COL_DIM))

    for _ in range(NUM_ITERATIONS):

        # Concatenate applys keras.All to the masks of all concatenated inputs
        #that means if one of them (initial_sequences) has a masked value, all of them will for the duration of the loop
        concat_sequences = layers.Concatenate()([initial_sequences, encoded_sequences, message_col_to_seq(columns, M)])
        encoded_sequences = seq_lstm(concat_sequences) #will always mask

        gathered_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(encoded_sequences, (-1, SITE_DIM) ))

        concat_columns = layers.Concatenate()([columns, message_seq_to_col(gathered_sequences, M)])
        columns = tf.squeeze(col_lstm(tf.expand_dims(concat_columns, axis=0)), axis=0)

        seq_concat = layers.Concatenate()([gathered_initial_sequences, gathered_sequences])
        M = mem_decoder([seq_concat, columns, sequence_lengths])

    relative_positions, gaps_start, gaps_in, gaps_end, col_dist = sec_decoder([M, columns, sequence_lengths])
    M_squared = tf.linalg.matmul(M, M, transpose_b=True)

    model = keras.Model(
        inputs=[sequences, sequence_gatherer, column_priors, sequence_lengths],
        outputs=[M_squared, relative_positions, gaps_start, gaps_in, gaps_end, col_dist],
    )

    model.compile(loss=["binary_crossentropy", "mse", "mse", "mse", "mse","categorical_crossentropy"],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss_weights=[MEM_LOSS, RP_LOSS, GAP_LOSS, GAP_LOSS, GAP_LOSS, COL_LOSS])

    return model
