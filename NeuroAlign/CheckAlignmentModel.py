import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import MSA
import AlignmentModel
import Postprocessing
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA


################################################################################################################################################
################################################################################################################################################

MAX_SEQ = 8

################################################################################################################################################
################################################################################################################################################

parser = argparse.ArgumentParser(description='Predicts a complete alignment given a fasta file with raw sequences.')
parser.add_argument("-f", type=str, default="", help="input fasta file")
parser.add_argument("-o", type=str, default="", help="output alignment fasta file")
parser.add_argument("-vis_mem", action='store_true', help="output a membership heatmap for max the first MAX_SEQ sequences and all iterations")
parser.add_argument("-vis_seq_rep", action='store_true', help="visualize the representations of each sequence site")
args = parser.parse_args()

################################################################################################################################################
################################################################################################################################################

al_model = AlignmentModel.make_model(training = False, output_representations = args.vis_seq_rep)
if os.path.isfile(AlignmentModel.CHECKPOINT_PATH+".index"):
    al_model.load_weights(AlignmentModel.CHECKPOINT_PATH)
else:
    print("Could not load alignment model.", flush=True)
    quit()

################################################################################################################################################
################################################################################################################################################

start_time = time.time()

msa = MSA.Instance(args.f, AlignmentModel.ALPHABET, gaps = True, contains_lower_case = True)

if not msa.valid:
    print("Invalid input file.", flush=True)
    quit()

maxlen = max(msa.seq_lens)
num_columns = max(msa.seq_lens)*AlignmentModel.COLUMN_OVEREXTENSION

#one-hot encode sequences
seq = np.zeros((len(msa.raw_seq), maxlen, len(AlignmentModel.ALPHABET)+2), dtype=np.float32)
for j,(l,si) in enumerate(zip(msa.seq_lens, msa.raw_seq)):
    lrange = np.arange(l)
    seq[j,lrange,si] = 1
    seq[j,lrange,len(AlignmentModel.ALPHABET)] = lrange / num_columns
    seq[j,lrange,len(AlignmentModel.ALPHABET)+1] = l / num_columns

#maps padded sequences to unpadded
sequence_gather_indices = np.concatenate([(i*maxlen + np.arange(l)) for i,l in enumerate(msa.seq_lens)], axis=0)
sequence_gather_indices = sequence_gather_indices.astype(np.int32)
rev_sequence_gather_indices = np.concatenate([(i*maxlen + np.flip(np.arange(l))) for i,l in enumerate(msa.seq_lens)], axis=0)
rev_sequence_gather_indices = rev_sequence_gather_indices.astype(np.int32)

initial_memberships = np.ones((sum(msa.seq_lens), int(num_columns))) / num_columns

input_dict = {  "sequences" : seq,
                "sequence_gather_indices" : sequence_gather_indices,
                "rev_sequence_gather_indices" : rev_sequence_gather_indices,
                "initial_memberships" : initial_memberships }

out = al_model(input_dict)

if args.vis_seq_rep:
    mem = [out[3*i] for i in range(int(len(out)/3))]
    seq_rep = [out[3*i+1] for i in range(int(len(out)/3))]
    cons_rep = [out[3*i+2] for i in range(int(len(out)/3))]
else:
    if AlignmentModel.NUM_ITERATIONS == 1:
        mem = [out]
    else:
        mem = out

cols = Postprocessing.seq_consistent(msa, mem[-1])

end_time = time.time()

print("prec, recall: ", msa.recall_prec(cols))
print("It took ", end_time - start_time, " seconds to construct the alignment.")

################################################################################################################################################
################################################################################################################################################

if args.o != "":
    MSA.column_pred_to_fasta(msa, cols, args.o)

################################################################################################################################################
################################################################################################################################################

def plot_heatmap_memberships(msa, mem):

    plt.subplots_adjust(hspace = .05, wspace=.05, left=0.05, right=0.95, top=0.98, bottom=0.02)

    ref = np.zeros((msa.membership_targets.size, msa.alignment_len))
    ref[np.arange(msa.membership_targets.size),msa.membership_targets] = 1
    cum_lens = np.cumsum(msa.seq_lens)
    ref_splits = np.split(ref, cum_lens[:-1])
    mem_splits = [np.split(m, cum_lens[:-1]) for m in mem]

    fontsize = 10

    for i, r  in enumerate(ref_splits):
        if i == MAX_SEQ:
            break
        ax = plt.subplot(min(len(msa.raw_seq), MAX_SEQ),AlignmentModel.NUM_ITERATIONS+1,i*(AlignmentModel.NUM_ITERATIONS+1)+1)
        if i == 0:
            ax.set_title("Reference", fontsize=fontsize)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_ylabel("Sequence {}".format(i+1), rotation=0, fontsize=fontsize, labelpad=22+fontsize)
        plt.imshow(r, cmap='hot', interpolation='nearest')

        for j in range(AlignmentModel.NUM_ITERATIONS):
            ax = plt.subplot(min(len(msa.raw_seq), MAX_SEQ),AlignmentModel.NUM_ITERATIONS+1,i*(AlignmentModel.NUM_ITERATIONS+1)+2+j)
            if i == 0:
                ax.set_title("Iteration {}".format(j+1), fontsize=fontsize)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(mem_splits[j][i], cmap='hot', interpolation='nearest')
    plt.show()

if args.vis_mem:
    plot_heatmap_memberships(msa, mem)

################################################################################################################################################
################################################################################################################################################

if args.vis_seq_rep:
    seq_rep = out[-2]
    consensus = out[-1]
    pca = PCA(n_components=2)
    seq_2d = pca.fit_transform(seq_rep)
    cmap = plt.cm.get_cmap('viridis', len(msa.seq_lens))
    colors = []
    for i,l in enumerate(msa.seq_lens):
        colors.extend([cmap(i)]*l)

    x_range = np.amax(seq_2d[:,0]) - np.amin(seq_2d[:,0])
    y_range = np.amax(seq_2d[:,1]) - np.amin(seq_2d[:,1])
    anno_offset = (x_range*0.004, y_range*0.004)

    #for s in msa.raw_seq:
        #colors.extend([cmap(aa) for aa in s])
    plt.scatter(seq_2d[:,0], seq_2d[:,1], color=colors)
    lsum = 0
    for s,l in zip(msa.raw_seq, msa.seq_lens):
        for i in range(l):
            aminoacid = AlignmentModel.ALPHABET[s[i]]
            plt.annotate(aminoacid, seq_2d[lsum+i]+anno_offset)
        lsum += l
    plt.show()
