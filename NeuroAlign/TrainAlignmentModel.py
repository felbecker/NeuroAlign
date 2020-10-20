import tensorflow as tf
import numpy as np
import random
import MSA
import AlignmentModel
import os.path
import time
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib as mpl
import matplotlib.pyplot as plt


##################################################################################################
##################################################################################################

pfam = ["PF"+"{0:0=5d}".format(i) for i in range(1,19228)]
pfam_not_found = 0

GPUS = tf.config.experimental.list_logical_devices('GPU')
NUM_DEVICES = max(1, len(GPUS))

if len(GPUS) > 0:
    print("Using " + str(NUM_DEVICES) + " GPU devices.", flush=True)
else:
    print("Using CPU.", flush=True)

##################################################################################################
##################################################################################################

#load reference alignments
msa = []

for f in pfam:
    try:
        m = MSA.Instance("Pfam/alignments/" + f + ".fasta", AlignmentModel.ALPHABET, gaps = True, contains_lower_case = True)
        #m = MSA.Instance("test/" + f, AlignmentModel.ALPHABET, gaps = True, contains_lower_case = True)
        msa.append(m)
        # if len(msa) == 4:
        #     break
    except FileNotFoundError:
        pfam_not_found += 1

np.random.seed(0)
random.seed(0)

indices = np.arange(len(msa))
np.random.shuffle(indices)
train, val = np.split(indices, [int(len(msa)*(1-AlignmentModel.VALIDATION_SPLIT))]) #np.array([0]), np.array([0])

##################################################################################################
##################################################################################################

#provides training batches
#each batch has an upper limit of sites
#sequences are drawn randomly from a protein family until all available sequences are chosen or
#the batch limit is exhausted.
class AlignmentBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, split, training = True):
        self.split = split
        #weights for random draw are such that large alignments that to not fit into a single batch
        #are drawn more often
        family_lens_total = [sum(msa[i].seq_lens) for i in self.split]
        family_weights = [max(1, t/AlignmentModel.BATCH_SIZE) for t in family_lens_total]
        sum_w = sum(family_weights)
        self.family_probs = [w/sum_w for w in family_weights]
        self.training = training

    def __len__(self):
        if self.training:
          return int(len(self.split)/3)
        else:
          return len(self.split)

    def sample_family(self, ext = ""):

        #draw a random family and sequences from that family
        family_i = np.random.choice(len(self.split), p=self.family_probs)
        family_m = msa[self.split[family_i]]
        seq_shuffled = list(range(len(family_m.raw_seq)))
        random.shuffle(seq_shuffled)
        batch_size = 0
        seqs_drawn = []
        for si in seq_shuffled:
            batch_size += family_m.seq_lens[si]
            if batch_size > AlignmentModel.BATCH_SIZE:
                if len(seqs_drawn) < 2:
                    #print("Batch size too small to fit at least 2 sequences. Resampling...")
                    return self.sample_family(ext)
                break
            else:
                seqs_drawn.append(si)
        batch_lens = [family_m.seq_lens[si] for si in seqs_drawn]

        maxlen = max(batch_lens)

        #one-hot encode sequences
        seq = np.zeros((len(seqs_drawn), maxlen, len(AlignmentModel.ALPHABET)), dtype=np.float32)
        for j,(l,si) in enumerate(zip(batch_lens, seqs_drawn)):
            lrange = np.arange(l)
            seq[j,lrange,family_m.raw_seq[si]] = 1

        #remove empty columns
        col_sizes = np.zeros(family_m.alignment_len)
        for j,(l, si) in enumerate(zip(batch_lens, seqs_drawn)):
            suml = sum(family_m.seq_lens[:si])
            col_sizes[family_m.membership_targets[suml:(suml+l)]] += 1
        empty = (col_sizes == 0)
        num_columns = np.sum(~empty)
        cum_cols = np.cumsum(empty)

        corrected_targets = []
        for j,(l, si) in enumerate(zip(batch_lens, seqs_drawn)):
            suml = sum(family_m.seq_lens[:si])
            ct = family_m.membership_targets[suml:(suml+l)]
            corrected_targets.append(ct - cum_cols[ct])

        #targets
        memberships = np.zeros((len(seqs_drawn), maxlen, num_columns), dtype=np.float32)
        for j,(l, targets) in enumerate(zip(batch_lens, corrected_targets)):
            r = np.arange(l)
            #memberships site <-> columns
            memberships[j,r,targets] = 1

        memberships = np.reshape(memberships, (-1, num_columns))

        sequence_gather_indices = np.concatenate([(i*maxlen + np.arange(l)) for i,l in enumerate(batch_lens)], axis=0)
        sequence_gather_indices = sequence_gather_indices.astype(np.int32)

        memberships = np.take(memberships, sequence_gather_indices, axis=0)
        memberships_sq = np.matmul(memberships, np.transpose(memberships))

        initial_memberships = np.ones((sum(batch_lens), num_columns)) / num_columns

        input_dict = {  ext+"sequences" : seq,
                        ext+"sequence_gather_indices" : sequence_gather_indices,
                        ext+"initial_memberships" : initial_memberships }
        target_dict = {ext+"mem"+str(i) : memberships_sq for i in range(AlignmentModel.NUM_ITERATIONS)}
        return input_dict, target_dict


    def __getitem__(self, index):
        if NUM_DEVICES == 1:
            return self.sample_family()
        else:
            input_dict, target_dict = {}, {}
            for i in range(NUM_DEVICES):
                id, td = self.sample_family("GPU_"+str(i)+"_")
                input_dict.update(id)
                target_dict.update(td)
            return input_dict, target_dict


train_gen = AlignmentBatchGenerator(train)
val_gen = AlignmentBatchGenerator(val, False)

##################################################################################################
##################################################################################################

def weighted_binary_crossentropy(y_true, y_pred):

        y_true = tf.reshape(y_true, (-1,1))
        y_pred = tf.reshape(y_pred, (-1,1))
        b_ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight_vector = tf.squeeze(y_true * AlignmentModel.POS_WEIGHT + (1. - y_true) * AlignmentModel.NEG_WEIGHT, -1)
        weighted_b_ce = weight_vector * b_ce
        return tf.keras.backend.sum(weighted_b_ce) / tf.keras.backend.sum(weight_vector)


##################################################################################################
##################################################################################################


al_model = AlignmentModel.make_model()
if os.path.isfile(AlignmentModel.CHECKPOINT_PATH+".index"):
    al_model.load_weights(AlignmentModel.CHECKPOINT_PATH)
    print("Loaded weights", flush=True)

if NUM_DEVICES == 1:
    model = al_model
    losses = {"mem"+str(i) : weighted_binary_crossentropy for i in range(AlignmentModel.NUM_ITERATIONS)}
    loss_weights = {"mem"+str(i) : 1 for i in range(AlignmentModel.NUM_ITERATIONS-1)}
    loss_weights["mem"+str(AlignmentModel.NUM_ITERATIONS-1)] = AlignmentModel.LAST_ITERATION_WEIGHT

    model.compile(loss=losses,
                  loss_weights = loss_weights,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=AlignmentModel.LEARNING_RATE),
                    metrics={"mem"+str(AlignmentModel.NUM_ITERATIONS-1) : [
                                    keras.metrics.TruePositives(name='tp'),
                                    keras.metrics.FalsePositives(name='fp'),
                                    keras.metrics.TrueNegatives(name='tn'),
                                    keras.metrics.FalseNegatives(name='fn'),
                                    keras.metrics.Precision(name='precision'),
                                    keras.metrics.Recall(name='recall')]})
else:
    inputs, all_outputs = [], []
    for i, gpu in enumerate(GPUS):
        with tf.device(gpu.name):
            sequences = keras.Input(shape=(None,AlignmentModel.SEQ_IN_DIM), name="GPU_"+str(i)+"_sequences")
            sequence_gather_indices = keras.Input(shape=(), name="GPU_"+str(i)+"_sequence_gather_indices", dtype=tf.int32)
            initial_memberships = keras.Input(shape=(None,), name="GPU_"+str(i)+"_initial_memberships")
            input_dict = {  "sequences" : sequences,
                            "sequence_gather_indices" : sequence_gather_indices,
                            "initial_memberships" : initial_memberships }
            outputs = al_model(input_dict)
            for j,o in enumerate(outputs):
                all_outputs.append(layers.Lambda(lambda x: x, name="GPU_"+str(i)+"_mem"+str(j))(o))
            inputs.extend([sequences, sequence_gather_indices, initial_memberships])

    model = keras.Model(inputs=inputs, outputs=all_outputs)
    losses = {}
    loss_weights = {}
    metrics = {}
    for i, gpu in enumerate(GPUS):
        losses.update({"GPU_"+str(i)+"_mem"+str(j) : weighted_binary_crossentropy for j in range(AlignmentModel.NUM_ITERATIONS)})
        loss_weights.update({"GPU_"+str(i)+"_mem"+str(j) : 1 for j in range(AlignmentModel.NUM_ITERATIONS-1)})
        loss_weights["GPU_"+str(i)+"_mem"+str(AlignmentModel.NUM_ITERATIONS-1)] = AlignmentModel.LAST_ITERATION_WEIGHT
        metrics.update({"GPU_"+str(i)+"_mem"+str(AlignmentModel.NUM_ITERATIONS-1) : [
                        keras.metrics.TruePositives(name='tp'),
                        keras.metrics.FalsePositives(name='fp'),
                        keras.metrics.TrueNegatives(name='tn'),
                        keras.metrics.FalseNegatives(name='fn'),
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall')]})

    model.compile(loss=losses, loss_weights=loss_weights,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=AlignmentModel.LEARNING_RATE),
                    metrics=metrics)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=AlignmentModel.CHECKPOINT_PATH,
                                                save_weights_only=True,
                                                verbose=1)

history = model.fit(train_gen,
            validation_data=val_gen,
            epochs = AlignmentModel.NUM_EPOCHS,
            verbose = 2,
            callbacks=[cp_callback])


##################################################################################################
##################################################################################################

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['figure.figsize'] = (16, 14)

def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
              color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
              color=colors[n], label='Val '+label,
  linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  plt.legend()

def plot_metrics(history):
  metrics =  ['loss',
              'mem'+str(AlignmentModel.NUM_ITERATIONS-1)+'_loss',
              'mem'+str(AlignmentModel.NUM_ITERATIONS-1)+'_precision',
              'mem'+str(AlignmentModel.NUM_ITERATIONS-1)+'_recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
              color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()



##################################################################################################
##################################################################################################

plot_loss(history, "NeuroAlign", 0)
plot_metrics(history)
plt.suptitle(AlignmentModel.CFG_TXT, fontsize=14)
plt.savefig(AlignmentModel.NAME + '.png')
