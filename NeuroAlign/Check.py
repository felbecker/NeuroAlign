import argparse

import MSA
import Model
from Config import config

parser = argparse.ArgumentParser(description='Checks the predictions of the latest trained model for a given test instance.')
parser.add_argument("-i", type=int, default=0, help="number of the input example to check")
parser.add_argument("-dir", type=str, default="./data_20", help="directory with data files")
args = parser.parse_args()

filepath = args.dir + "/A"+"{0:0=4d}".format(args.i)+".fa"
msa = MSA.Instance(filepath)

# print(msa.nodes)
# print(msa.membership_targets)

predictor = Model.NeuroAlignPredictor(config, msa)
predictor.load_latest()
n_rp, c_rp, s_n, s_c = predictor.predict(msa, msa.alignment_len)

print(n_rp)
print(c_rp)
print(s_n)
print(s_c)
