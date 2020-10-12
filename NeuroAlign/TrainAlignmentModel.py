import tensorflow as tf
import numpy as np
import random
import MSA
import AlignmentModel
import os.path
import time


pfam = ["PF"+"{0:0=5d}".format(i) for i in range(1,19228)]
pfam_not_found = 0

##################################################################################################
##################################################################################################

#load reference alignments
msa = []

for f in pfam:
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
train, val = np.split(indices, [int(len(msa)*(1-AlignmentModel.VALIDATION_SPLIT))])

##################################################################################################
##################################################################################################

#provides training batches
#each batch has an upper limit of sites
#sequences are drawn randomly from a protein family until all available sequences are chosen or
#the batch limit is exhausted.
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
        return len(self.split) #steps per epoch

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
                    #print("Batch size too small to fit at least 2 sequences. Resampling...")
                    return self.__getitem__(index)
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

        #maps padded sequences to unpadded
        sequence_gatherer = np.zeros((sum(batch_lens), len(seqs_drawn)*maxlen), dtype=np.float32)
        suml = 0
        for i,l in enumerate(batch_lens):
            sequence_gatherer[np.arange(suml, suml+l), np.arange(i*maxlen, i*maxlen+l)] = 1
            suml += l

        memberships = np.matmul(sequence_gatherer, memberships)
        memberships_sq = np.matmul(memberships, np.transpose(memberships))

        initial_memberships = np.ones((sum(batch_lens), num_columns)) / num_columns

        input_dict = {  "sequences" : seq,
                        "sequence_gatherer" : sequence_gatherer,
                        "initial_memberships" : initial_memberships }
        target_dict = {"mem"+str(i) : memberships for i in range(AlignmentModel.NUM_ITERATIONS)}
        return input_dict, target_dict


train_gen = AlignmentBatchGenerator(train)
val_gen = AlignmentBatchGenerator(val)

##################################################################################################
##################################################################################################

model = AlignmentModel.make_model()
if os.path.isfile(AlignmentModel.CHECKPOINT_PATH+".index"):
    model.load_weights(AlignmentModel.CHECKPOINT_PATH)
    print("Loaded weights", flush=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=AlignmentModel.CHECKPOINT_PATH,
                                                save_weights_only=True,
                                                verbose=1)

model.fit(train_gen,
            validation_data=val_gen,
            epochs = AlignmentModel.NUM_EPOCHS,
            verbose = 2,
            callbacks=[cp_callback])
