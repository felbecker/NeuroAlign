import argparse
import MSA
import Model
import numpy as np
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

from Config import config
import Postprocessing

parser = argparse.ArgumentParser(description='Tests the latest NeuroAlign model.')
parser.add_argument("-n", type=int, default=200, help="number of testing examples")
parser.add_argument("-dir", type=str, default="./data_20_test", help="directory with data files")
parser.add_argument("-w", action='store_true', help="write output fasta files")
args = parser.parse_args()

if args.w:
    try:
        os.rmdir("./test_out")
    except OSError:
        print ("Can not remove test directory")
    try:
        os.mkdir("./test_out")
    except OSError:
        print ("Can not remove test directory")

#load the training dataset
msa = []
alphabet = ['A', 'C', 'G', 'T'] if config["type"] == "nucleotide" else ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X']
for i in range(1,args.n+1):
    filepath = args.dir + "/A"+"{0:0=4d}".format(i)+".fa"
    inst = MSA.Instance(filepath, alphabet)
    if inst.valid:
        msa.append(inst)

#instantiate the predictor
predictor = Model.NeuroAlignPredictor(config, msa[0])
predictor.load_latest()

ps_ml = 0
rs_ml = 0
ps_gc = 0
rs_gc = 0
for m in msa:
    mem, rp, gaps = predictor.predict(m)
    am_ml = Postprocessing.greedy_col_max_likely(m, mem)
    am_gc = Postprocessing.greedy_consistent(m, mem)

    print("___________________________")
    print("file:", m.filename)
    print("target sequences: \n", m.ref_seq)
    print("___________________________")
    print("target memberships:", m.membership_targets)
    print("greedy fast:", am_ml)
    print("greedy consistent:", am_gc)
    print("___________________________")
    print("target rp:", m.membership_targets/m.alignment_len)
    print("predicted rp:", rp)
    print("___________________________")
    print("target gaps:", m.gap_lengths)
    print("predicted gaps:", gaps)
    print("___________________________")

    p,r = m.recall_prec(am_ml.flatten())
    ps_ml += p
    rs_ml += r
    p,r = m.recall_prec(am_gc.flatten())
    ps_gc += p
    rs_gc += r

    if args.w:
        MSA.column_pred_to_fasta(m, am_ml, "./test_out/")

print("max likely precision=", ps_ml/len(msa), "recall=", rs_ml/len(msa))
print("greedy consistent precision=", ps_gc/len(msa), "recall=", rs_gc/len(msa))
