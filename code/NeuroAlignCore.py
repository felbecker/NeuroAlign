from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
from graph_nets import modules
from graph_nets import blocks

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import copy
import sys
sys.path.append('./ProcessSeq')

import PatternSet

def get_region_nodes_input(pattern_set):
    #node inputs are relative positions in sequences and relative lengths
    return [[np.float32(i/pattern_set.anchor_set.len_seqs[int(seq)]),
                np.float32(l/pattern_set.anchor_set.len_seqs[int(seq)]),
                np.float32(score/PatternSet.MAX_PAIRWISE_SCORE)]
                        for (seq,i,l),score in zip(pattern_set.region_node_list, pattern_set.region_scores)]

def get_pattern_nodes_input(pattern_set):
    pattern_nodes_input = []
    sum_inst = 0
    for num, score in zip(pattern_set.num_instances_per_pattern, pattern_set.pattern_scores):
        pattern_nodes_input.extend([[np.float32(i/num),
                                    np.float32(pattern_set.region_node_list[sum_inst][2]/pattern_set.anchor_set.len_seqs[int(pattern_set.region_node_list[sum_inst][0])]),
                                    np.float32(score/PatternSet.MAX_PAIRWISE_SCORE)] for i in range(num)])
        sum_inst += num
    return pattern_nodes_input


#utility method that converts a given pattern set to sequence, pattern and target dictsZZ
def pattern_set_to_input_target_dicts(pattern_set):

    #node inputs are relative positions in sequences and relative lengths
    seq_g_nodes = get_region_nodes_input(pattern_set) + get_pattern_nodes_input(pattern_set)
    #seq_g_nodes = np.ndarray(shape=(len(seq_g_nodes),len(seq_g_nodes[0])), buffer=seq_g_nodes, dtype=np.float32)

    #print("seqs:", pattern_set.anchor_set.len_seqs)
    #print("region nodes:", pattern_set.region_node_list)
    #print("SEQ G NODES:", seq_g_nodes)


    nodes_sequence_wise = []
    for s in range(pattern_set.anchor_set.num_seq):
        nodes_on_seq = [i for i in range(len(pattern_set.region_node_list))
                                if pattern_set.region_node_list[i][0] == s]
        nodes_on_seq.sort(key=lambda x: pattern_set.region_node_list[x][1])
        nodes_sequence_wise.append(nodes_on_seq)

    seq_g_senders = []
    seq_g_receivers = []
    gap_rel_length = []
    for s in nodes_sequence_wise:
        for i in range(1,len(s)):
            seq_g_senders.append(s[i-1])
            seq_g_receivers.append(s[i])
            (seq,pos1,len1) = pattern_set.region_node_list[s[i-1]]
            (_,pos2,len2) = pattern_set.region_node_list[s[i]]
            len_seq = pattern_set.anchor_set.len_seqs[int(seq)]
            gap_rel_length.append((pos2 - pos1 - len1)/len_seq) #relative length of the gap between region i-1 and iZ

    gap_rel_length = np.reshape(np.array(gap_rel_length, dtype=np.float32), (-1,1))

    sequence_dict = {
        "globals": [np.float32(0)],
        "nodes": seq_g_nodes,
        "edges": gap_rel_length,
        "senders": seq_g_senders,
        "receivers": seq_g_receivers
    }




    pattern_g_senders = []
    pattern_g_receivers = []
    pattern_edges = []
    pattern_edge_targets = []
    sum_inst = len(pattern_set.region_node_list)
    region_index = 0

    #1D edges will be distributed as follows: (P = pattern, I = Instance, R = region)
    #P1[R1[I1,I2,..], R2[I1,I2,..], ...], P2[...], ...
    for num_region, num_inst in zip(pattern_set.num_regions_per_pattern, pattern_set.num_instances_per_pattern):
        for i in range(num_region):

            pattern_g_senders.extend([region_index]*num_inst)
            pattern_g_receivers.extend(range(sum_inst, sum_inst+num_inst))
            pattern_edges.extend([1/num_inst]*num_inst)
            pattern_edge_targets.extend(
                        [1 if pattern_set.instance_membership_targets[region_index] == n else 0
                            for n in range(num_inst)])

            pattern_g_senders.extend(range(sum_inst, sum_inst+num_inst))
            pattern_g_receivers.extend([region_index]*num_inst)
            pattern_edges.extend([1/num_inst]*num_inst)
            pattern_edge_targets.extend(
                        [1 if pattern_set.instance_membership_targets[region_index] == n else 0
                            for n in range(num_inst)])

            region_index += 1
        sum_inst += num_inst

    pattern_edges = np.reshape(np.array(pattern_edges, dtyppattern_edgese=np.float32), (-1,1))
    pattern_edge_targets = np.reshape(np.array(pattern_edge_targets, dtype=np.float32), (-1,1))

    pattern_dicts = {
        "globals": [np.float32(0)],
        "nodes": seq_g_nodes,#np.zeros((len(seq_g_nodes),1), dtype=np.float32),
        "edges": pattern_edges,
        "senders": pattern_g_senders,
        "receivers": pattern_g_receivers
    }

    assert pattern_set.target_initialized, "call compute_targets(pattern_set) first"

    target_dict = {
        "globals": [np.float32(0)],
        "nodes": np.concatenate([np.reshape(pattern_set.region_rp_targets.astype(np.float32), (-1,1)),
                                    np.reshape(pattern_set.pattern_rp_targets.astype(np.float32), (-1,1))], axis=0),
        "edges": pattern_edge_targets,
        "senders": pattern_g_senders,
        "receivers": pattern_g_receivers
    }target_dict

    return sequence_dict, pattern_dicts, target_dicts




def make_mlp(layersizes):
    if len(layersizes)>0:
        return lambda: snt.Sequential([
                snt.nets.MLP(layersizes, activate_final=True),
                snt.LayerNorm()])
    else:
        return None

class MLPGraphIndependent(snt.AbstractModule):
    def __init__(self,
                edge_fn_layer_sizes = [],  #can be set to an empty list to ignore edges / nodes / globals
                node_fn_layer_sizes = [],
                global_fn_layer_sizes = [],
                name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
          self._network = modules.GraphIndependent(
              edge_model_fn=make_mlp(edge_fn_layer_sizes),
              node_model_fn=make_mlp(node_fn_layer_sizes),
              global_model_fn=make_mlp(global_fn_layer_sizes))

    def _build(self, inputs):
        return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
    def __init__(self, node_layer_s, edge_layer_s, globals_layer_s, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphNetwork(node_model_fn=make_mlp(node_layer_s),
                                           edge_model_fn=make_mlp(edge_layer_s),
                                           global_model_fn=make_mlp(globals_layer_s),
                                           reducer = tf.math.unsorted_segment_mean)

    def _build(self, inputs):
        return self._network(inputs)



#
# given a set of high scoring region pairs with "non overlap property"
# in a set of protein sequences
#
# denote the following graph as "AL-Graph":
# a vertex v_i for each region i; attributes: (pos, length)
# forward edges for subsequent regions in each sequence v_i -> v_i+1
#
# in addition to the AL graph, consider the following set of nodes:
# vertices p_j_s for each pattern j; s=1,...,"numpattern_j"
#
# One iteration of the ProteinGNN consists of the following steps:
# 1) 1 Iteration on the region graph [ALGNN]
# 2) send updated node tensors to pattern nodes (melatent0 = latentan) [deep set]
# 3) 1 Iteration on the complete pattern graph [fully connected GNN]
# 4) send updated pattern tensors to region nodes ["inverse deep set" ??]
# 5) send updated node tensors to sequence nodes (sum?) [deep set]
#
#
# INPUT: a sequence graph, region nodes labeled with relative positions in the raw sequences
# OUTPUT: a sequence graph with updated relative node positions in the final alignment
#
#latent0 = latent
class ProteinGNN(snt.AbstractModule):

    def __init__(self,
                param, #a dictionary that maps strings to low level parameters
                name="ProteinGNN"):

        super(ProteinGNN, self).__init__(name=name)

        #encodes the input regions
        self.sequence_encoder = MLPGraphIndependent(node_fn_layer_sizes=param["enc_node_layer_s"],
                                                    edge_fn_layer_sizes=param["enc_edge_layer_s"],
                                                    global_fn_layer_sizes=param["enc_globals_layer_s"])

        #encodes the pattern input
        self.pattern_encoder = MLPGraphIndependent(node_fn_layer_sizes=param["enc_node_layer_s"],
                                                    edge_fn_layer_sizes=param["enc_edge_layer_s"],
                                                    global_fn_layer_sizes=param["enc_globals_layer_s"])

        #core graph network that operates on the sequence graph
        self.sequence_graph_core = MLPGraphNetwork(node_layer_s=param["seq_core_node_layer_s"],
                                                   edge_layer_s=param["seq_core_edge_layer_s"],
                                                   globals_layer_s=param["seq_core_globals_layer_s"])

        #foward network to map from region nodes in the seq graph to pattern instance nodes
        self.pattern_graph_core = MLPGraphNetwork(node_layer_s = param["pattern_core_node_layer_s"],
                                                edge_layer_s = param["pattern_core_edge_layer_s"],
                                                globals_layer_s = param["pattern_core_globals_layer_s"])

        #decodes the latent tensors of the region nodes to the actual output
        self.decoder = MLPGraphIndependent(node_fn_layer_sizes=param["enc_node_layer_s"],
                                           edge_fn_layer_sizes=param["enc_edge_layer_s"])

        self.output_transform = modules.GraphIndependent(
                                            node_model_fn = lambda: snt.Linear(1, name="node_output"),
                                            edge_model_fn = lambda: snt.Linear(1, name="edge_output"))



    #
    def _build(self,
                sequence_graph, #graphs tuple with a single graph for the sequence graph
                pattern_graph, #graphs tuple with a single graph for each pattern
                num_processing_steps):

        latent_seq_graph = self.sequence_encoder(sequence_graph)
        latent_seq_graph0 = latent_seq_graph

        latent_pattern_graph = self.pattern_encoder(pattern_graph)
        latent_pattern_graph0 = latent_pattern_graph

        for _ in range(num_processing_steps):

            #core input is the concatenation of original input graph and the graph from the last iteration
            seq_core_input = utils_tf.concat([latent_seq_graph0, latent_seq_graph], axis=1)

            #the latent representation of the input graph in the current iteration
            latent_seq_graph = self.sequence_graph_core(seq_core_input)

            #copy node tensors from the latent sequence graph to the pattern graph
            latent_pattern_graph = latent_pattern_graph.replace(nodes = latent_seq_graph.nodes)
            pattern_core_input = utils_tf.concat([latent_pattern_graph0, latent_pattern_graph], axis=1)
            latent_pattern_graph = self.pattern_graph_core(pattern_core_input)

            latent_seq_graph = latent_seq_graph.replace(nodes = latent_pattern_graph.nodes)


        decoded_op = self.decoder(latent_pattern_graph)
        return self.output_transform(decoded_op)




class Predictor():

    def __init__(self, param, example_seq_input, example_pattern_input, example_target):

        self.model = ProteinGNN(param)

        self.seq_input_ph = utils_tf.placeholders_from_data_dicts([example_seq_input])
        self.pattern_input_ph = utils_tf.placeholders_from_data_dicts([example_pattern_input])
        self.target_ph = utils_tf.placeholders_from_data_dicts([example_target])

        self.train_logits = self.model(self.seq_input_ph, self.pattern_input_ph, param["train_mp_iterations"])
        self.test_logits = self.model(self.seq_input_ph, self.pattern_input_ph, param["test_mp_iterations"])


        def create_loss():

            loss_op = tf.math.reduce_sum( tf.losses.softmax_cross_entropy() )

            return loss_op

        self.loss_train = create_loss(self.train_logits)
        self.loss_test = create_loss(self.test_logits)

        optimizer = tf.train.MomentumOptimizer(param["learning_rate"], param["optimizer_momentum"])
        self.step_op = optimizer.minimize(self.loss_train)

        self.seq_input_ph_run = utils_tf.make_runnable_in_session(self.seq_input_ph)
        self.pattern_input_ph_run = utils_tf.make_runnable_in_session(self.pattern_input_ph)
        self.target_ph_run = utils_tf.make_runnable_in_session(self.target_ph)

        self.train_out = tf.sigmoid(self.train_logits.edges)
        self.test_out = tf.sigmoid(self.test_logits.edges)


    def make_feed_dict(self, seqIn, patternIn, target=None):
        seqIn = utils_np.data_dicts_to_graphs_tuple(seqIn)
        patternIn = utils_np.data_dicts_to_graphs_tuple(patternIn)
        feed_dict = utils_tf.get_feed_dict( self.seq_input_ph, seqIn )
        feed_dict.update( utils_tf.get_feed_dict( self.pattern_input_ph, patternIn ) )
        if not target == None:
            target = utils_np.data_dicts_to_graphs_tuple(target)
            feed_dict.update( utils_tf.get_feed_dict( self.target_ph, target ) )
        return feed_dict


    def train(self, session, seqIn, patternIn, target):
        loss, _ = session.run([self.loss_train, self.step_op],
                feed_dict = self.make_feed_dict(seqIn, patternIn, target))
        return loss


    def test(self, session, test_seq_graphs, test_pattern_graphs, target_graphs):
        test_loss_sum = 0
        for seqIn, patternIn, target in zip(test_seq_graphs, test_pattern_graphs, target_graphs):
            test_loss = session.run([self.loss_test], feed_dict = self.make_feed_dict([seqIn], [patternIn], [target]))
            test_loss_sum += test_loss[0]
        return test_loss_sum/len(test_seq_graphs)


    def predict(self, session, seqIn, patternIn):
        out = session.run([self.test_out], feed_dict = self.make_feed_dict([seqIn], [patternIn]))
        return out[0]
