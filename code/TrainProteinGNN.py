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

parser = argparse.ArgumentParser(description='Training')
parser.add_argument("-r", type=int, default=10, help="the kmer radius")
parser.add_argument("-steps_so_far", type=int, default=0, help="stepcount to resume training")
parser.add_argument("-batch_size", type=int, default=10000, help="cap for the sum of graph nodes in a batch")
parser.add_argument("-t", type=int, default=0, help="treshold")
parser.add_argument("-a", type=int, default=200, help="maximum number of anchors allowed")
args = parser.parse_args()


num_alignments = 1509
data_threads = 20
test_frac = 0.2
num_training_iteration = 10000
savestate_milestones = 100

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)
random.seed(SEED)

print("Data reading and preprocessing. This may take some time...")

def load_one(filename):
    #print("Reading ", filename)
    anchor_set = AnchorSet.anchor_set_from_file(filename)
    AnchorSet.read_solution("data/"+filename.split('/')[1]+".fasta", anchor_set)
    pattern_set = PatternSet.find_patterns(anchor_set)
    PatternSet.compute_targets(pattern_set)
    seq_graph, pattern_graph, target_graph = ProteinGraphNN.pattern_set_to_input_target_dicts(pattern_set)
    return seq_graph, pattern_graph, target_graph

filenames = []
for i in range(num_alignments):
    #files can be missing when no anchors were found for a parameter choice r, t, minrow
    #skip these files
    dir = "anchors_"+str(args.r)+"_"+str(args.t)+"_"+str(args.a)
    if os.path.isfile(dir+"/A"+"{0:0=4d}".format(i)+".anchors") and os.path.isfile(dir+"/A"+"{0:0=4d}".format(i)+".meta"):
        filenames.append(dir+"/A"+"{0:0=4d}".format(i))


#data
pool = multiprocessing.Pool(data_threads)
result = pool.map(load_one, filenames)
result = [r for r in result]
result = list(zip(result, filenames))
random.shuffle(result)
data, filenames = zip(*result)
data, filenames = list(data), list(filenames)
input_seq_graphs, input_pattern_graphs, target_graphs = zip(*data)
input_seq_graphs, input_pattern_graphs, target_graphs = list(input_seq_graphs), list(input_pattern_graphs), list(target_graphs)
num_test_instances = max(1,int(np.floor(len(input_seq_graphs)*test_frac)))

#remember training/test split
os.makedirs(os.path.dirname("./model_proteinGNN"), exist_ok=True)
with open("./model_proteinGNN/traing_instances.txt", "w") as f:
    for filename in filenames[num_test_instances:]:
        f.write(filename+"\n")
with open("./model_proteinGNN/test_instances.txt", "w") as f:
    for filename in filenames[:num_test_instances]:
        f.write(filename+"\n")

def split(graphs):
    train_graphs = graphs[num_test_instances:]
    test_graphs = graphs[:num_test_instances]
    return train_graphs, test_graphs

train_seq_input_graphs, test_seq_input_graphs = split(input_seq_graphs)
train_pattern_input_graphs, test_pattern_input_graphs = split(input_pattern_graphs)
train_target_graphs, test_target_graphs = split(target_graphs)


tf.reset_default_graph()
predictor = ProteinGraphNN.Predictor(ModelParameters.param,
                                     train_seq_input_graphs[0],
                                     train_pattern_input_graphs[0],
                                     train_target_graphs[0])

session = tf.Session()
session.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=1000)

outf = open("train_out", 'w')

def print__(*args):
    textline = " ".join([str(a) for a in args])
    outf.write(textline+"\n")
    outf.flush()
    print(textline)

if args.steps_so_far > 0:
    saver.restore(session, "./model_proteinGNN/it_"+str(args.steps_so_far)+".ckpt")
    print("Successfully loaded model num ", args.steps_so_far)

batch_losses = []

for i in range(args.steps_so_far, args.steps_so_far + num_training_iteration):

    seq_in_batch = []
    pat_in_batch = []
    tar_batch = []
    #randomly draw samples until batch is full (allow overflow; batch_size = #edges)
    while sum([len(g["nodes"]) for g in seq_in_batch]) < args.batch_size:
        choice = np.random.randint(len(train_seq_input_graphs))
        seq_in_batch.append(train_seq_input_graphs[choice])
        pat_in_batch.append(train_pattern_input_graphs[choice])
        tar_batch.append(train_target_graphs[choice])

    batch_losses.append(predictor.train(session, seq_in_batch, pat_in_batch, tar_batch))

    if len(batch_losses) < 100:
        print__("Iteration ", i, " with batch loss ", batch_losses[-1])
    else:
        print__("Iteration ", i, " with batch loss ", batch_losses[-1], " Av last 100:", sum(batch_losses[-100:])/100)

    if i%savestate_milestones==0 and i>0:
        saver.save(session, "./model_proteinGNN/it_"+str(i)+".ckpt")

        test_loss = predictor.test(session, test_seq_input_graphs, test_pattern_input_graphs, test_target_graphs)
        print__("Test loss: ", test_loss)

outf.close()
