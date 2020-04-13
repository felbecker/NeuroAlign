import argparse
import MSA
import Model
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from Config import config

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

ps = 0
rs = 0
for m in msa:
    _,_,_,mpc = predictor.predict(m, m.alignment_len)
    am = np.argmax(mpc, axis=1)
    print(am)
    p,r = m.recall_prec(am.flatten())
    ps += p
    rs += r

print("precision=", ps/len(msa), "recall=", rs/len(msa))
