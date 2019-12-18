#third party imports
import tensorflow as tf

from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
from graph_nets import modules
from graph_nets import blocks

import matplotlib.pyplot as plt
import networkx as nx
import argparse
import multiprocessing
import os
import random
import numpy as np

#project imports
import ProteinGraphNN
import ProcessSeq

parser = argparse.ArgumentParser(description='Computes edge sets of alignment graphs for all balibase ref alignments')
parser.add_argument("-r", type=int, default=10, help="the kmer radius used for training")
parser.add_argument("-model_steps", type=int, default=0, help="stepcount of the model to load")
args = parser.parse_args()

#threads for data loading
data_threads = 4

#number of message passing iterations testing
test_mp_iterations = 50

print("Data reading and preprocessing. This may take some time...")

max_score, max_len = ProcessSeq.load_meta_data("anchors_"+str(args.r)+"/meta.txt")

filenames = []
with open("./model_proteinGNN/test_instances.txt", "r") as f:
    for line in f:
        filenames.append(line.strip())

anchor_set = ProcessSeq.anchor_set_from_file(filenames[0])
local_inconsistencies = ProcessSeq.compute_local_inconsistencies(anchor_set)
example_input_dict, example_target_dict = ProteinGraphNN.anchor_set_to_input_target_dicts(anchor_set, local_inconsistencies, max_score, max_len)

#load a previously trained model and make predictions
tf.reset_default_graph()
graphNN = ProteinGraphNN.ProteinGNN(0.1, 10, test_mp_iterations, example_input_dict, example_target_dict)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "./model_proteinGNN/it_"+str(args.model_steps)+".ckpt")


for f in filenames:
    print("testing ", f)
    anchor_set = ProcessSeq.anchor_set_from_file(f)
    local_inconsistencies = ProcessSeq.compute_local_inconsistencies(anchor_set)
    input_dict, target_dict = ProteinGraphNN.anchor_set_to_input_target_dicts(anchor_set, local_inconsistencies, max_score, max_len)
    input_graph = utils_np.data_dicts_to_graphs_tuple([input_dict])
    prediction = graphNN.predict(sess, input_graph)
    greedy_selection = ProcessSeq.greedy_best(anchor_set, prediction[0])
    data = anchor_set.anchor_data[np.array(greedy_selection, dtype=bool)]
    #anchor data is stored as a float matrix (to be contiguous in memory)
    #however, for the output, everything except the score is converted to integers
    name = f.split('/')[1]
    with open("./neuroalign_pred/"+name+".anc", "w") as f:
        line = ""
        for i in range(data.shape[0]):
            line += str(int(data[i,ProcessSeq.SEQ_ST])+1) + "\t" #dialigns sequences start with 1
            line += str(int(data[i,ProcessSeq.SEQ_END])+1) + "\t"
            line += str(int(data[i,ProcessSeq.IND_ST])) + "\t"
            line += str(int(data[i,ProcessSeq.IND_END])) + "\t"
            line += str(int(data[i,ProcessSeq.LEN])) + "\t"
            line += str(data[i,ProcessSeq.SCORE]) + "\n"
        f.write(line)
