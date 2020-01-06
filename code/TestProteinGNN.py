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
import ModelParameters
import sys
sys.path.append('./ProcessSeq')
import AnchorSet
import PatternSet

parser = argparse.ArgumentParser(description='Computes edge sets of alignment graphs for all balibase ref alignments')
parser.add_argument("-r", type=int, default=10, help="the kmer radius used for training")
parser.add_argument("-model_steps", type=int, default=0, help="stepcount of the model to load")
parser.add_argument("-draw", type=int, default=0, help="if 1, each prediction is also rendered for debugging; comps are paused while a plot is open")
parser.add_argument("-t", type=int, default=0, help="treshold")
parser.add_argument("-a", type=int, default=200, help="maximum number of anchors allowed")
parser.add_argument("-start_at", type=int, default=0, help="starting test instances, skips previous")
args = parser.parse_args()

#threads for data loading
data_threads = 4

#number of message passing iterations testing
test_mp_iterations = 30

print("Data reading and preprocessing. This may take some time...")

filenames = []
with open("../data/model_proteinGNN/test_instances.txt", "r") as f:
    for line in f:
        filenames.append(line.strip())

#load a previously trained model and make predictions
anchor_set = AnchorSet.anchor_set_from_file("../data/"+filenames[0])
AnchorSet.read_solution("../data/data/"+filenames[0].split('/')[1]+".fasta", anchor_set)
pattern_set = PatternSet.find_patterns(anchor_set)
PatternSet.compute_targets(pattern_set)
example_seq_graph, example_pattern_graph, example_target_graph = ProteinGraphNN.pattern_set_to_input_target_dicts(pattern_set)


predictor = ProteinGraphNN.Predictor(ModelParameters.param,
                                    example_seq_graph,
                                    example_pattern_graph,
                                    example_target_graph)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "../data/model_proteinGNN/it_"+str(args.model_steps)+".ckpt")

global_pred = np.zeros((0))
global_ref = np.zeros((0))
for f in filenames[args.start_at:]:
    print("testing ", f)
    anchor_set = AnchorSet.anchor_set_from_file("../data/"+f)
    AnchorSet.read_solution("../data/data/"+f.split('/')[1]+".fasta", anchor_set)
    pattern_set = PatternSet.find_patterns(anchor_set)
    PatternSet.compute_targets(pattern_set)
    seq_graph, pattern_graph, target_graph = ProteinGraphNN.pattern_set_to_input_target_dicts(pattern_set)
    rppred, _ = predictor.predict(sess, seq_graph, pattern_graph)

    prediction = PatternSet.get_anchor_probs(pattern_set, rppred[:len(pattern_set.region_node_list)])
    greedy_selection = AnchorSet.greedy_best(anchor_set, prediction)
    unwrapped_selection = AnchorSet.unwrap_selection(anchor_set, greedy_selection)
    unwrapped_anchors = AnchorSet.unwrap_anchor_set(anchor_set)
    name = f.split('/')[1]
    AnchorSet.read_solution("../data/data/"+name+".fasta", unwrapped_anchors)
    reference_solution = unwrapped_anchors.solution
    score_pred = AnchorSet.jaccard_index(unwrapped_selection, reference_solution)
    print("prediction score: ", score_pred)

    global_pred = np.concatenate((global_pred, unwrapped_selection), axis=0)
    global_ref = np.concatenate((global_ref, reference_solution), axis=0)

    if args.draw == 1:
        PatternSet.attach_prediction(pattern_set, rppred[:len(pattern_set.region_node_list)], rppred[len(pattern_set.region_node_list):])
        fig = plt.figure(1, figsize=(20, 6))
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        pattern_set.draw_pattern_graph(ax)
        plt.title('score:'+str(score_pred))
        plt.draw()
        plt.show()

print("global prediction score:", AnchorSet.jaccard_index(global_pred, global_ref))
