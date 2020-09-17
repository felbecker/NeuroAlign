import MSA
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Prints basic properties of a given fasta file.')
parser.add_argument("-f", type=str, default="", help="fasta file")
args = parser.parse_args()


ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'O', 'U']


#load the data
msa = MSA.Instance(args.f, ALPHABET, gaps=False)

if not msa.valid:
    print("Invalid data.")
else:
    print("number of sequences: ", len(msa.seq_lens))
    minlen = np.argmin(msa.seq_lens)
    maxlen = np.argmax(msa.seq_lens)
    print("min seq length: ", msa.seq_lens[minlen], "(", msa.seq_ids[minlen] ,")")
    print("max seq length: ", msa.seq_lens[maxlen], "(", msa.seq_ids[maxlen] ,")")
