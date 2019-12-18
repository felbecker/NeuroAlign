from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import ProcessSeq

#CONFIGURATION
num_processing_steps_tr = 3
num_processing_steps_ge = 3

def alignment_to_graph_dict(nodes, edges, priors):

    nodes = [[0.0] for n in nodes]
    edges = 

    data_dict = {
        "nodes": nodes,
        "edges": instance.edge_types,
        "senders": instance.senders,
        "receivers": instance.receivers
    }

    return data_dict
