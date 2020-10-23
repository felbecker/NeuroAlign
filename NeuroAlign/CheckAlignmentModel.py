import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import MSA
import AlignmentModel
import Postprocessing
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Predicts a complete alignment given a fasta file with raw sequences.')
parser.add_argument("-f", type=str, default="", help="input fasta file")
parser.add_argument("-o", type=str, default="", help="output alignment fasta file")
args = parser.parse_args()

al_model = AlignmentModel.make_model(training = False)
if os.path.isfile(AlignmentModel.CHECKPOINT_PATH+".index"):
    al_model.load_weights(AlignmentModel.CHECKPOINT_PATH)
else:
    print("Could not load alignment model.", flush=True)
    quit()

msa = MSA.Instance(args.f, AlignmentModel.ALPHABET, gaps = True, contains_lower_case = True)

if not msa.valid:
    print("Invalid input file.", flush=True)
    quit()

maxlen = max(msa.seq_lens)

#one-hot encode sequences
seq = np.zeros((len(msa.raw_seq), maxlen, len(AlignmentModel.ALPHABET)), dtype=np.float32)
for j,(l,si) in enumerate(zip(msa.seq_lens, msa.raw_seq)):
    lrange = np.arange(l)
    seq[j,lrange,si] = 1

#maps padded sequences to unpadded
sequence_gather_indices = np.concatenate([(i*maxlen + np.arange(l)) for i,l in enumerate(msa.seq_lens)], axis=0)
sequence_gather_indices = sequence_gather_indices.astype(np.int32)

num_columns = max(msa.seq_lens)*2

initial_memberships = np.ones((sum(msa.seq_lens), int(num_columns))) / num_columns

input_dict = {  "sequences" : seq,
                "sequence_gather_indices" : sequence_gather_indices,
                "initial_memberships" : initial_memberships }

mem = al_model(input_dict)

cols = Postprocessing.seq_consistent(msa, mem[-1])
print("prec, recall: ", msa.recall_prec(cols))
#MSA.column_pred_to_fasta(msa, cols, args.o)

def plot_heatmap_memberships(msa, mem):

    plt.subplots_adjust(hspace = .05, wspace=.05, left=0.05, right=0.95, top=0.98, bottom=0.02)

    ref = np.zeros((msa.membership_targets.size, msa.alignment_len))
    ref[np.arange(msa.membership_targets.size),msa.membership_targets] = 1
    cum_lens = np.cumsum(msa.seq_lens)
    ref_splits = np.split(ref, cum_lens[:-1])
    mem_splits = [np.split(m, cum_lens[:-1]) for m in mem]

    fontsize = 10

    for i, r  in enumerate(ref_splits):
        ax = plt.subplot(len(msa.raw_seq),AlignmentModel.NUM_ITERATIONS+1,i*(AlignmentModel.NUM_ITERATIONS+1)+1)
        if i == 0:
            ax.set_title("Reference", fontsize=fontsize)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_ylabel("Sequence {}".format(i+1), rotation=0, fontsize=fontsize, labelpad=22+fontsize)
        plt.imshow(r, cmap='hot', interpolation='nearest')

        for j in range(AlignmentModel.NUM_ITERATIONS):
            ax = plt.subplot(len(msa.raw_seq),AlignmentModel.NUM_ITERATIONS+1,i*(AlignmentModel.NUM_ITERATIONS+1)+2+j)
            if i == 0:
                ax.set_title("Iteration {}".format(j+1), fontsize=fontsize)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(mem_splits[j][i], cmap='hot', interpolation='nearest')
    plt.show()

#plot_heatmap_memberships(msa, mem)
