import argparse

import MSA
import Model
from Config import config

parser = argparse.ArgumentParser(description='Trains and tests a NeuroAlign model for the simple case of exact nucleotide matches.')
parser.add_argument("-i", type=int, default=0, help="number of the input example to check")
parser.add_argument("-dir", type=str, default="./data", help="directory with data files")
args = parser.parse_args()

filepath = args.dir + "/A"+"{0:0=4d}".format(args.i)+".fa"
msa = MSA.Instance(filepath)

# print(msa.nodes)
# print(msa.membership_targets)

predictor = Model.NeuroAlignPredictor(config, msa)
n_rp, c_rp, mem = predictor.predict(msa, 4)

print(n_rp)
print(c_rp)
print(mem)
