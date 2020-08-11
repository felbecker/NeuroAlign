import argparse
import MSA
import Model
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from Config import config
import Postprocessing

parser = argparse.ArgumentParser(description='Tests the latest NeuroAlign model.')
parser.add_argument("-n", type=int, default=200, help="number of testing examples")
parser.add_argument("-dir", type=str, default="./data_20_test", help="directory with data files")
args = parser.parse_args()

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
    print("___________________________")
    print(m.filename)
    print(m.ref_seq)
    mem, rp, gaps = predictor.predict(m)
    print(m.membership_targets)
    am_ml = Postprocessing.greedy_col_max_likely(m, mem)
    am_gc = Postprocessing.greedy_consistent(m, mem)
    print(am_ml)
    print(am_gc)
    p,r = m.recall_prec(am_ml.flatten())
    ps_ml += p
    rs_ml += r
    p,r = m.recall_prec(am_gc.flatten())
    ps_gc += p
    rs_gc += r

print("max likely precision=", ps_ml/len(msa), "recall=", rs_ml/len(msa))
print("greedy consistent precision=", ps_gc/len(msa), "recall=", rs_gc/len(msa))
