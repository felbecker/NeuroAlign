from Bio import Phylo
import MSA
import DistanceModel
import numpy as np
import SequenceModel
import os.path
import tensorflow as tf


pfam = ["PF"+"{0:0=5d}".format(i) for i in range(1,19228)]
pfam_not_found = 0

##################################################################################################
##################################################################################################

#load the trees
trees = []

#load the sequences (just remove the gaps from pfam alignments)
msa = []

for f in pfam:
    try:
        m = MSA.Instance("Pfam/alignments/" + f + ".fasta", DistanceModel.ALPHABET, gaps = False)
        if len(m.raw_seq) > 1:
            msa.append(m)
            tree = Phylo.read("Pfam/trees/" + f + ".tree", "newick")
            trees.append(tree)
    except FileNotFoundError:
        pfam_not_found += 1

np.random.seed(0)

indices = np.arange(len(msa))
np.random.shuffle(indices)

print("Traing on ", len(msa), " families.")
print(sum([len(m.raw_seq) for m in msa]), " sequences in total.")

train, val = np.split(indices, [int(len(msa)*(1-DistanceModel.VALIDATION_SPLIT))])

##################################################################################################
##################################################################################################

sequence_model = SequenceModel.make_model()
if os.path.isfile(SequenceModel.CHECKPOINT_PATH+".index"):
    sequence_model.load_weights(SequenceModel.CHECKPOINT_PATH)
    sequence_model.layers.pop() #pop time distributed dense
    sequence_model.layers.pop() #pop softmax
else:
    print("No sequence model found.")
    #quit()

##################################################################################################
##################################################################################################

#samples random batches of BATCH_SIZE pairs of sequences and a vector of
#length BATCH_SIZE with corresponding phylogenetic target distances
class DistanceBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, split):
        self.split = split
        self.num_seq_total = sum([len(msa[i].raw_seq) for i in self.split])
        self.family_probs = [len(msa[i].raw_seq)/self.num_seq_total for i in self.split]

    #randomly draws a pair of sequences and their corresponding phylogenetic distance
    def sample_one(self):
        sample_i = np.random.choice(len(self.split), p=self.family_probs)
        m, t = msa[self.split[sample_i]], trees[self.split[sample_i]]
        #assume that msa and tree contain the same set of sequences, if not, fix data
        s1, s2 = np.random.randint(len(m.raw_seq), size=2)
        while s2 == s1:
            s2 = np.random.randint(len(m.raw_seq))
        try:
            dist = t.distance(m.seq_ids[s1], m.seq_ids[s2])
        except ValueError as err:
            print(t)
            print(t.find_clades(m.seq_ids[s1]))
            print(t.find_clades(m.seq_ids[s2]))
            print("ValueError: {0}".format(err))
            print(m.seq_ids)
            print(m.filename, m.seq_ids[s1], m.seq_ids[s2])
        return m.raw_seq[s1], m.raw_seq[s2], dist

    def __len__(self):
        return int(np.floor(len(self.split) / DistanceModel.BATCH_SIZE)) #steps per epoch

    def __getitem__(self, index):
        seq1 = []
        seq2 = []
        dists = np.zeros((DistanceModel.BATCH_SIZE,1), dtype=np.float32)
        for i in range(DistanceModel.BATCH_SIZE):
            s1, s2, d = self.sample_one()
            seq1.append(s1)
            seq2.append(s2)
            dists[i] = d
        lens1 = [s.shape[0] for s in seq1]
        lens2 = [s.shape[0] for s in seq2]
        num_steps = max(lens1+lens2)
        seq = np.zeros((2*DistanceModel.BATCH_SIZE, num_steps, len(DistanceModel.ALPHABET)), dtype=np.float32)
        for j,(l1,l2,s1,s2) in enumerate(zip(lens1, lens2, seq1, seq2)):
            seq[2*j,np.arange(l1),s1] = 1
            seq[2*j+1,np.arange(l2),s2] = 1
        #embedded_seq = sequence_model(seq)
        #return np.concatenate((seq, embedded_seq), axis=2), dists
        return seq, dists

train_gen = DistanceBatchGenerator(train)
val_gen = DistanceBatchGenerator(val)

##################################################################################################
##################################################################################################

model = DistanceModel.make_model()
if os.path.isfile(DistanceModel.CHECKPOINT_PATH+"/model.ckpt.index"):
    model.load_weights(DistanceModel.CHECKPOINT_PATH)
    print("Loaded weights.", flush=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=DistanceModel.CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
model.fit(train_gen,
            validation_data=val_gen,
            epochs = DistanceModel.NUM_EPOCHS,
            verbose = 1,
            callbacks=[cp_callback])
