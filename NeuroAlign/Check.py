import argparse
import tensorflow as tf
import graph_nets as gn

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
n_rp, c_rp, rel_occ, mpc = predictor.predict(msa, msa.alignment_len)

seq_g, col_g = predictor._graphs_from_instance(msa, msa.alignment_len)

print("seq input:", seq_g)
print("col inputs:", col_g)
print("_________________________________________________________")

print("node relative pos pred:", n_rp)
print("node relative pos target:", msa.node_rp_targets)
print("_________________________________________________________")

print("col relative pos pred:", c_rp)
print("col relative pos target:", col_g.globals)
print("_________________________________________________________")

print("relative occ per col pred:", rel_occ)
print("relative occ per col target:", msa.rel_occ_per_column)
print("_________________________________________________________")

print("memberships pred:", mpc)
print("memberships target:", tf.one_hot(msa.membership_targets, gn.utils_tf.get_num_graphs(col_g)))
