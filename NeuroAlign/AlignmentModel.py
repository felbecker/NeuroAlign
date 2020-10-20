import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']

NUM_ITERATIONS = 3
SITE_DIM = 64
COL_DIM = 128
SEQ_LSTM_DIM = 256
COL_LSTM_DIM = 256
ENCODER_LAYERS = [128]
COL_MSGR_LAYERS = [128,128]
SEQ_MSGR_LAYERS = [128,128]
DECODER_LAYERS = [64]
SHARED_ITERATIONS = False #if false, each message passing iteration uses unique parameters

VALIDATION_SPLIT = 0.01

#maximum number of sites in a batch
#must be at least as large as the sum of the two longest sequences in all families
BATCH_SIZE = 5000
COL_BATCHES = 1
LEARNING_RATE = 2e-4
MEM_LOSS = 1
RP_LOSS = 1
GAP_LOSS = 1
COL_LOSS = 0.1
NUM_EPOCHS = 200
DROPOUT = 0.0

CHECKPOINT_PATH = "alignment_checkpoints/model.ckpt"


print("iterations: ", NUM_ITERATIONS,
      " site_dim: ", SITE_DIM,
      " col_dim: ", COL_DIM,
      " seq_lstm: ", SEQ_LSTM_DIM,
      " col_lstm: ", COL_LSTM_DIM,
      " encoder: ", ENCODER_LAYERS,
      "col_msg: ", COL_MSGR_LAYERS,
      "seq_msg: ", SEQ_MSGR_LAYERS,
      "dec: ", DECODER_LAYERS,
      "mem_loss: ", MEM_LOSS,
      "rp_loss: ", RP_LOSS,
      "gap_loss: ", GAP_LOSS,
      "col_loss: ", COL_LOSS,
      "batch: ", BATCH_SIZE,
      "learning_rate: ", LEARNING_RATE, flush=True)

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

    def call(self, inputs):
        if type(inputs) is not list and len(inputs) != 3:
            raise Exception('MembershipDecoder must be called on a list of exactly 3 tensors ')
        seq_dec = self.decoder_s(inputs[0])
        col_dec = self.decoder_c(inputs[1])
        sequence_lengths = inputs[2]

        n_batch_size = tf.cast(tf.math.ceil(tf.cast(tf.shape(col_dec)[0], dtype=tf.float32)/COL_BATCHES), dtype=tf.int32)
        logits = []
        for i in range(COL_BATCHES):
            split = col_dec[(i*n_batch_size):((i+1)*n_batch_size)]
            s_logits = tf.expand_dims(seq_dec, 1) + tf.expand_dims(split, 0) #use broadcasting to efficiently get all combinations
            s_logits = self.decoder_relu(tf.reshape(s_logits, (-1, DECODER_LAYERS[0])))
            s_logits = self.decoder(layers.Dropout(DROPOUT)(s_logits))
            s_logits = self.out_trans(s_logits)
            logits.append(tf.reshape(s_logits, (-1, tf.shape(split)[0])))
        M = self.decoder_sm(tf.concat(logits, axis=1))
        return M

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
    seq_in_dim = len(ALPHABET)
    sequences = keras.Input(shape=(None,seq_in_dim), name="sequences")
    sequence_lengths = keras.Input(shape=(), name="sequence_lengths", dtype=tf.int32)
    sequence_gatherer = keras.Input(shape=(None,), name="sequence_gatherer")
    column_priors = keras.Input(shape=(None,), name="column_priors")

    encoder = MLP(ENCODER_LAYERS+[SITE_DIM])
    #networks
    if SHARED_ITERATIONS:
        seq_lstm = [layers.LSTM(SEQ_LSTM_DIM, return_sequences=True, dropout = DROPOUT)]*NUM_ITERATIONS
        seq_dense_time_dist = [layers.TimeDistributed(layers.Dense(SITE_DIM))]*NUM_ITERATIONS
        seq_layer_norm_time_dist = [layers.TimeDistributed(layers.LayerNormalization())]*NUM_ITERATIONS
        col_lstm = [layers.LSTM(COL_LSTM_DIM, return_sequences=True, dropout = DROPOUT)]*NUM_ITERATIONS
        col_dense_time_dist = [layers.TimeDistributed(layers.Dense(COL_DIM))]*NUM_ITERATIONS
        col_layer_norm_time_dist = [layers.TimeDistributed(layers.LayerNormalization())]*NUM_ITERATIONS
        col_messenger = [MLP(COL_MSGR_LAYERS)]*NUM_ITERATIONS
        seq_messenger = [MLP(SEQ_MSGR_LAYERS)]*NUM_ITERATIONS
        mem_decoder = [MembershipDecoder()]*NUM_ITERATIONS
        sec_decoder = [SecDecoder()]*NUM_ITERATIONS
    else:
        seq_lstm = [layers.LSTM(SEQ_LSTM_DIM, return_sequences=True, dropout = DROPOUT) for _ in range(NUM_ITERATIONS)]
        seq_dense_time_dist = [layers.TimeDistributed(layers.Dense(SITE_DIM)) for _ in range(NUM_ITERATIONS)]
        seq_layer_norm_time_dist = [layers.TimeDistributed(layers.LayerNormalization()) for _ in range(NUM_ITERATIONS)]
        col_lstm = [layers.LSTM(COL_LSTM_DIM, return_sequences=True, dropout = DROPOUT) for _ in range(NUM_ITERATIONS)]
        col_dense_time_dist = [layers.TimeDistributed(layers.Dense(COL_DIM)) for _ in range(NUM_ITERATIONS)]
        col_layer_norm_time_dist = [layers.TimeDistributed(layers.LayerNormalization()) for _ in range(NUM_ITERATIONS)]
        col_messenger = [MLP(COL_MSGR_LAYERS) for _ in range(NUM_ITERATIONS)]
        seq_messenger = [MLP(SEQ_MSGR_LAYERS) for _ in range(NUM_ITERATIONS)]
        mem_decoder = [MembershipDecoder() for _ in range(NUM_ITERATIONS)]
        sec_decoder = [SecDecoder() for _ in range(NUM_ITERATIONS)]

    def message_seq_to_col(gathered_sequences, M, i):
        messages_from_seqs = seq_messenger[i](gathered_sequences)
        messages_to_cols = tf.matmul(M, messages_from_seqs, transpose_a = True)
        return messages_to_cols

    def message_col_to_seq(columns, M, i):
        messages_from_cols = col_messenger[i](columns)
        messages_to_seqs = tf.linalg.matmul(M, messages_from_cols)
        messages_to_seqs = tf.linalg.matmul(sequence_gatherer, messages_to_seqs, transpose_a = True)
        return tf.reshape(messages_to_seqs, (tf.shape(sequences)[0], tf.shape(sequences)[1], COL_MSGR_LAYERS[-1]))

    #encode the sequences
    masked_sequences = layers.Masking(mask_value=0.0)(sequences)
    encoded_sequences = layers.TimeDistributed(encoder)(masked_sequences)
    gathered_initial_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(sequences, (-1, seq_in_dim) ))

    #initial memberships
    M = column_priors

    #initial columns
    columns = tf.ones((tf.shape(M)[1], COL_DIM))

    mem_sq_out = []
    rp_out = []
    gaps_start_out = []
    gaps_in_out = []
    gaps_end_out = []
    col_dist_out = []

    for i in range(NUM_ITERATIONS):

        # Concatenate applys keras.All to the masks of all concatenated inputs
        #that means if one of them (initial_sequences) has a masked value, all of them will for the duration of the loop
        concat_sequences = layers.Concatenate()([sequences, encoded_sequences, message_col_to_seq(columns, M, i)])
        encoded_sequences = seq_dense_time_dist[i](seq_lstm[i](concat_sequences)) #will always mask
        encoded_sequences = seq_layer_norm_time_dist[i](encoded_sequences)

        gathered_sequences = tf.linalg.matmul(sequence_gatherer, tf.reshape(encoded_sequences, (-1, SITE_DIM) ))
        seq_concat = layers.Concatenate()([gathered_initial_sequences, gathered_sequences])

        concat_columns = layers.Concatenate()([columns, message_seq_to_col(seq_concat, M, i)])
        columns = col_dense_time_dist[i](col_lstm[i](tf.expand_dims(concat_columns, axis=0)))
        columns = col_layer_norm_time_dist[i](columns)
        columns = tf.squeeze(columns, axis=0)

        M = mem_decoder[i]([seq_concat, columns, sequence_lengths])

        relative_positions, gaps_start, gaps_in, gaps_end, col_dist = sec_decoder[i]([M, columns, sequence_lengths])
        M_squared = tf.linalg.matmul(M, M, transpose_b=True)

        #name outputs by passing to identity lambdas..
        mem_sq_out.append(layers.Lambda(lambda x: x, name="mem"+str(i))(M_squared))
        rp_out.append(layers.Lambda(lambda x: x, name="rp"+str(i))(relative_positions))
        gaps_start_out.append(layers.Lambda(lambda x: x, name="gs"+str(i))(gaps_start))
        gaps_in_out.append(layers.Lambda(lambda x: x, name="g"+str(i))(gaps_in))
        gaps_end_out.append(layers.Lambda(lambda x: x, name="ge"+str(i))(gaps_end))
        col_dist_out.append(layers.Lambda(lambda x: x, name="col"+str(i))(col_dist))

    model = keras.Model(
        inputs=[sequences, sequence_gatherer, column_priors, sequence_lengths],
        outputs=mem_sq_out + rp_out + gaps_start_out + gaps_in_out + gaps_end_out + col_dist_out,
    )

    losses = {"mem"+str(i) : "binary_crossentropy" for i in range(NUM_ITERATIONS)}
    losses.update({"rp"+str(i) : "mse" for i in range(NUM_ITERATIONS)})
    losses.update({"gs"+str(i) : "mse" for i in range(NUM_ITERATIONS)})
    losses.update({"g"+str(i) : "mse" for i in range(NUM_ITERATIONS)})
    losses.update({"ge"+str(i) : "mse" for i in range(NUM_ITERATIONS)})
    losses.update({"col"+str(i) : "categorical_crossentropy" for i in range(NUM_ITERATIONS)})

    loss_weights = {"mem"+str(i) : MEM_LOSS for i in range(NUM_ITERATIONS)}
    loss_weights.update({"rp"+str(i) : RP_LOSS for i in range(NUM_ITERATIONS)})
    loss_weights.update({"gs"+str(i) : GAP_LOSS for i in range(NUM_ITERATIONS)})
    loss_weights.update({"g"+str(i) : GAP_LOSS for i in range(NUM_ITERATIONS)})
    loss_weights.update({"ge"+str(i) : GAP_LOSS for i in range(NUM_ITERATIONS)})
    loss_weights.update({"col"+str(i) : COL_LOSS for i in range(NUM_ITERATIONS)})

    model.compile(loss=losses,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss_weights=loss_weights,
                    metrics={"mem"+str(NUM_ITERATIONS-1) : [keras.metrics.Precision(), keras.metrics.Recall()]})

    return model
