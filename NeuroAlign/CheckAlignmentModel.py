import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import MSA
import AlignmentModel
import Postprocessing
import argparse
import numpy as np

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

msa = MSA.Instance(args.f, AlignmentModel.ALPHABET, gaps = False, contains_lower_case = True)

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

num_columns = max(msa.seq_lens)*1.05

initial_memberships = np.ones((sum(msa.seq_lens), int(num_columns))) / num_columns

input_dict = {  "sequences" : seq,
                "sequence_gather_indices" : sequence_gather_indices,
                "initial_memberships" : initial_memberships }

mem = al_model(input_dict)[-1]

cols = Postprocessing.seq_consistent(msa, mem)
np.set_printoptions(threshold=np.inf)
print(cols)
#MSA.column_pred_to_fasta(msa, cols, args.o)
