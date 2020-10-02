import tensorflow as tf
import numpy as np
import random
import MSA
import AlignmentModel
import os.path
import time


USE_GPU = False

pfam = ["PF"+"{0:0=5d}".format(i) for i in range(1,19228)]
pfam_not_found = 0

##################################################################################################
##################################################################################################

#load reference alignments
msa = []

for f in pfam[:1]:
    try:
        m = MSA.Instance("Pfam/alignments/" + f + ".fasta", AlignmentModel.ALPHABET, gaps = True, contains_lower_case = True)
        #m = MSA.Instance("test/" + f, AlignmentModel.ALPHABET, gaps = True, contains_lower_case = True)
        msa.append(m)
    except FileNotFoundError:
        pfam_not_found += 1

np.random.seed(0)
random.seed(0)

indices = np.arange(len(msa))
np.random.shuffle(indices)
train, val = np.array([0]),np.array([0])#np.split(indices, [int(len(msa)*(1-AlignmentModel.VALIDATION_SPLIT))])


##################################################################################################
##################################################################################################

#provides training batches
#each batch has an upper limit of sites
#sequences are drawn randomly from a protein family until all available sequences are chosen or
#the batch limit is exhausted. An error is thrown, if the batch limit is such that less than 2 sequences fit into it
class AlignmentBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, split):
        self.split = split
        #weights for random draw are such that large alignments that to not fit into a single batch
        #are drawn more often
        family_lens_total = [sum(msa[i].seq_lens) for i in self.split]
        family_weights = [max(1, t/AlignmentModel.BATCH_SIZE) for t in family_lens_total]
        sum_w = sum(family_weights)
        self.family_probs = [w/sum_w for w in family_weights]

    def __len__(self):
        return 50#len(self.split) #steps per epoch

    def __getitem__(self, index):

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
                    raise Exception("Batch size too small to fit at least 2 sequences.")
                break
            else:
                seqs_drawn.append(si)
        batch_lens = [family_m.seq_lens[si] for si in seqs_drawn]

        maxlen = max(batch_lens)

        #one-hot encode sequences + relative positions
        seq = np.zeros((len(seqs_drawn), maxlen, len(AlignmentModel.ALPHABET)+2), dtype=np.float32)
        for j,(l,si) in enumerate(zip(batch_lens, seqs_drawn)):
            lrange = np.arange(l)
            seq[j,lrange,family_m.raw_seq[si]] = 1
            seq[j,lrange,len(AlignmentModel.ALPHABET)] = (lrange+1)/l
            seq[j,lrange,len(AlignmentModel.ALPHABET)+1] = (lrange+1)/family_m.alignment_len

        #targets
        memberships = np.zeros((len(seqs_drawn), maxlen, family_m.alignment_len), dtype=np.float32)
        relative_positions = np.zeros((len(seqs_drawn), maxlen), dtype=np.float32)
        col_dists = np.zeros((family_m.alignment_len, len(AlignmentModel.ALPHABET)+1))
        gaps_in = np.zeros((sum(batch_lens)-len(batch_lens), 1))
        gaps_start = np.zeros((len(seqs_drawn)))
        gaps_end = np.zeros((len(seqs_drawn)))
        for j,(l, si) in enumerate(zip(batch_lens, seqs_drawn)):
            suml = sum(family_m.seq_lens[:si])
            targets = family_m.membership_targets[suml:(suml+l)]
            r = np.arange(l)

            #memberships site <-> columns
            memberships[j,r,targets] = 1

            #relative positions of sequence nodes
            relative_positions[j,r] = targets/family_m.alignment_len

            #aminoacid-distribution for each column
            col_dists[targets, family_m.raw_seq[si]] += 1

            #inner gap lengths between sites
            gaps_in[r[:-1] + sum(batch_lens[:j]) - j, 0] = family_m.gap_lengths[(suml+si+1):(suml+si+l)] / family_m.alignment_len
            gaps_start[j] = family_m.gap_lengths[suml+si] / family_m.alignment_len
            gaps_end[j] = family_m.gap_lengths[suml+si+l] / family_m.alignment_len

        col_dists[:,len(AlignmentModel.ALPHABET)] = len(seqs_drawn) - np.sum(col_dists, axis=1)
        col_dists /= len(seqs_drawn)

        memberships = np.reshape(memberships, (-1, family_m.alignment_len))

        #maps padded sequences to unpadded
        sequence_gatherer = np.zeros((sum(batch_lens), len(seqs_drawn)*maxlen), dtype=np.float32)
        suml = 0
        for i,l in enumerate(batch_lens):
            sequence_gatherer[np.arange(suml, suml+l), np.arange(i*maxlen, i*maxlen+l)] = 1
            suml += l

        # #initial column memberships
        M_s = np.ones((len(seqs_drawn), max(batch_lens), family_m.alignment_len), dtype=np.float32)
        M_s /= family_m.alignment_len #sum over columns = 1
        M_c = np.ones_like(M_s)
        M_c /= np.reshape(np.array(batch_lens), (-1,1,1)) #sum over sequence = 1
        column_priors = M_s + M_c - M_s * M_c

        column_priors = np.matmul(sequence_gatherer, np.reshape(column_priors, (-1, family_m.alignment_len)))
        memberships = np.matmul(sequence_gatherer, memberships)
        relative_positions = np.matmul(sequence_gatherer, np.reshape(relative_positions, (-1, 1)))

        input_dict = {  "sequences" : seq,
                        "sequence_gatherer" : sequence_gatherer,
                        "column_priors" : column_priors,
                        "sequence_lengths" : np.array(batch_lens, dtype=np.int32) }
        target_dict = {"mem" : np.matmul(memberships, np.transpose(memberships)),
                        "rp" : relative_positions,
                        "gs" : gaps_start,
                        "g" : gaps_in,
                        "ge" : gaps_end,
                        "col" : col_dists }
        return input_dict, target_dict


train_gen = AlignmentBatchGenerator(train)
val_gen = AlignmentBatchGenerator(val)

##################################################################################################
##################################################################################################

def make_model():
    model = AlignmentModel.make_model()
    if os.path.isfile(AlignmentModel.CHECKPOINT_PATH+".index"):
        model = tf.keras.models.load_weights(AlignmentModel.CHECKPOINT_PATH)
        print("Loaded weights", flush=True)
    return model

if USE_GPU:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync), flush=True)
    with strategy.scope():
        model = make_model()
else:
    model = make_model()


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=AlignmentModel.CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_gen,
            #validation_data=val_gen,
            epochs = AlignmentModel.NUM_EPOCHS,
            verbose = 2,
            callbacks=[cp_callback])


#input, target = train_gen.__getitem__(0)

#print(input, target)

#automatic model visualization... does not work especially well with many message passing iterations
#tf.keras.utils.plot_model(model, "NeuroAlign.png", show_shapes=True)
#
# start = time.time()
# out = model(input)
# end = time.time()
#print(out)
# print(end-start)
