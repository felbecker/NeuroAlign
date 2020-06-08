import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import numpy as np
import os


#
#all involved neural networks are MLPs with layer sizes defined in Config.py
#each layer is relu activated
#layer normalization is applied after the last layer
#
def make_mlp_model(layersizes):
    return lambda: snt.Sequential([
        snt.nets.MLP(layersizes, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])

def get_len_alphabet(config):
    return 4 if config["type"] == "nucleotide" else 23

def init_weights(shape):
    return np.random.normal(0, 1, shape).astype(dtype=np.float32)

#
# a module that learns a representation for each symbol in the underlying alphabet (e.g. proteins)
# and for their interactions (pairwise and higher order)
# the initial representation (before message passing) is a fully parameterized, complete graph
# with parameter vectors of size D for each node, edge and for the global tensor, where D can be set in Config.py
# (for simplicity everthing has the same hidden dimension, but this can be finetuned)
# We exploit the fact that the alphabet size is statically known: Each letter and each interaction (edge) has its
# own parameters.
#
# The learned representation of the alphabet communicates with the representation of the sequences by
# sending a message per symbol to positions representing the respective symbol. It can receive information
# from the sequence by reducing all messages from sequence positions of the same symbol.
#
# inputs: cur_alphabet_graph - the current representation of the alphabet graph
#         messages_from_seq - messages from each sequence position
#         messages_from_columns - messages from the representation of each sequence position of each column
#         seq_indices - integer tensor specifying the index of the respective member of the alphabet for each sequence position
#
# output: a tensor containing a message for each sequence position from the respective member of the alphabet
#
class AlphabetKernel(snt.Module):
    def __init__(self, config, name = "AlphabetKernel"):

        super(AlphabetKernel, self).__init__(name=name)

        self.config = config
        self.lenA = get_len_alphabet(config)

        self.node_param = tf.Variable(
                                    init_weights([self.lenA, config["alphabet_latent_dim"]]),
                                        trainable=True, name="alphabet_node_param")

        self.edge_param = tf.Variable(
                                    init_weights([self.lenA*self.lenA, config["alphabet_latent_dim"]]),
                                        trainable=True, name="alphabet_edge_param")

        self.global_param = tf.Variable(
                                    tf.zeros([config["alphabet_latent_dim"]]),
                                        trainable=True, name="alphabet_global_param")

        self.alphabet_param_graph = gn.utils_tf.data_dicts_to_graphs_tuple([{
                                "nodes" : tf.zeros_like(self.node_param),
                                "globals" : tf.zeros_like(self.global_param) }])
        self.alphabet_param_graph = gn.utils_tf.fully_connect_graph_static(self.alphabet_param_graph)
        self.alphabet_param_graph = self.alphabet_param_graph.replace(
                                        nodes = self.node_param, edges=self.edge_param, globals = tf.reshape(self.global_param, (1, -1)))

        self.alphabet_network = gn.modules.GraphNetwork(
                                    edge_model_fn=make_mlp_model(config["alphabet_net_edge_layers"]+[config["alphabet_latent_dim"]]),
                                    node_model_fn=make_mlp_model(config["alphabet_net_node_layers"]+[2*config["alphabet_latent_dim"]]),
                                    global_model_fn=make_mlp_model(config["alphabet_net_global_layers"]+[config["alphabet_latent_dim"]]),
                                    reducer = tf.math.unsorted_segment_sum)

        self.sequence_messenger = make_mlp_model(config["alphabet_to_sequence_layers"]+[config["seq_latent_dim"]])()
        self.column_messenger = make_mlp_model(config["alphabet_to_column_layers"]+[config["col_latent_dim"]])()



    def __call__(self, cur_alphabet_graph, messages_from_seq, messages_from_columns, seq_indices, num_col, num_pos):

        #compute group means for each symbol over all corresponding positions in the sequences
        segments = tf.tile(tf.range(num_pos), [num_col])
        messages_from_columns = tf.math.unsorted_segment_sum(messages_from_columns, segments, num_pos)
        messages_concat = tf.math.unsorted_segment_mean(tf.concat([messages_from_seq, messages_from_columns], axis=1), seq_indices, self.lenA)

        #concatenate current states, the messages and the initial state of the alphabet for stability
        cur_alphabet_graph = cur_alphabet_graph.replace(nodes = tf.concat([cur_alphabet_graph.nodes, messages_concat], axis=1))
        cur_alphabet_graph = gn.utils_tf.concat([cur_alphabet_graph, self.alphabet_param_graph], axis = 1)

        #update
        cur_alphabet_graph = self.alphabet_network(cur_alphabet_graph)

        #compute messages to sequences and columns
        messages_out = cur_alphabet_graph.nodes[:, self.config["alphabet_latent_dim"]:]
        messages_to_seq = self.sequence_messenger(messages_out)
        messages_to_columns = self.column_messenger(messages_out)

        #cut the messages from the alphabet graph
        cur_alphabet_graph = cur_alphabet_graph.replace(nodes = cur_alphabet_graph.nodes[:, :self.config["alphabet_latent_dim"]])

        return cur_alphabet_graph, messages_to_seq, messages_to_columns



#
# A module that utilizes a deep set to predict the contents of each column, based
# on messages from each sequence position, a column specific latent tensor for each sequence
# position and a global tensor for each sequence position.
#
# Columns are initialized as follows:
# The column specific states for each sequence position must be initialized by a prior that states
# a probability of participation to the column for each sequence position.
# The global representation for each column is initialized by a shared parameter vector similar to the parameterization
# of the alphabet graph.
#
# input: initial_column_graph - initial encoding of the col priors used for stability
#        cur_column_graph - current representation of each column
#        messages_from_seq - messages coming from each position in the sequencen
#        messages_from_alphabet - messages from each alphabet symbol
#
# output: updated representation of the columns
#         messages to be send to every sequence node
#
class ColumnKernel(snt.Module):
    def __init__(self, config, name = "ColumnKernel"):

        super(ColumnKernel, self).__init__(name=name)

        self.config = config
        self.lenA = get_len_alphabet(config)

        self.column_network = gn.modules.DeepSets(
                                    node_model_fn=make_mlp_model(config["column_net_node_layers"]+[2*config["col_latent_dim"]]),
                                    global_model_fn=make_mlp_model(config["column_net_global_layers"]+[config["col_latent_dim"]]),
                                    reducer = tf.math.unsorted_segment_sum)

        self.col_global_param = tf.Variable(
                                    tf.zeros([1, config["col_latent_dim"]]),
                                        trainable=True, name="col_global_param")

        self.col_node_param = tf.Variable(
                                    tf.zeros([1, config["col_latent_dim"]]),
                                        trainable=True, name="col_node_param")

        self.sequence_messenger = make_mlp_model(config["alphabet_to_sequence_layers"]+[config["seq_latent_dim"]])()
        self.alphabet_messenger = make_mlp_model(config["alphabet_to_column_layers"]+[config["alphabet_latent_dim"]])()

        #self.column_encoder = gn.modules.GraphIndependent(node_model_fn = make_mlp_model(config["column_encode_node_layers"] + [config["col_latent_dim"]]))

        self.column_decoder = gn.modules.GraphIndependent(node_model_fn=make_mlp_model(config["column_decode_node_layers"] + [config["col_latent_dim"]]))
        self.column_out_transform = gn.modules.GraphIndependent(node_model_fn = lambda: snt.Linear(1, name="column_out_transform"))



    def encode(self, col_priors):
        #cur_column_graph = self.column_encoder(col_priors)
        return col_priors.replace(nodes = tf.tile(self.col_node_param, [gn.utils_tf.get_num_graphs(col_priors)*col_priors.n_node[0],1]),
                                globals = tf.tile(self.col_global_param, [gn.utils_tf.get_num_graphs(col_priors),1]))



    def __call__(self, memberships, initial_column_graph, cur_column_graph, messages_from_alphabet, messages_from_seq, seq_indices):

        n_g = gn.utils_tf.get_num_graphs(cur_column_graph)
        n_n = cur_column_graph.n_node[0]

        #tile the input messages accordingly
        messages_from_alphabet = tf.matmul(tf.one_hot(seq_indices, self.lenA), messages_from_alphabet)
        messages_concat = tf.tile(tf.concat([messages_from_seq, messages_from_alphabet], axis = 1), [n_g, 1])

        #for better indel prediction, we concatenate with the states of adjacent columns
        null_state = tf.zeros_like(cur_column_graph.nodes[:n_n,:], dtype=tf.float32)
        prev_cols = tf.concat([null_state, cur_column_graph.nodes[:-n_n,:]] ,axis=0)
        nxt_cols = tf.concat([cur_column_graph.nodes[n_n:,:], null_state] ,axis=0)
        null_global = tf.zeros_like(cur_column_graph.globals[0:1,:], dtype=tf.float32)
        prev_globals = tf.concat([null_global, cur_column_graph.globals[:-1,:]], axis=0)
        nxt_globals = tf.concat([cur_column_graph.globals[1:,:], null_global], axis=0)
        cur_column_graph = cur_column_graph.replace(nodes = tf.concat([prev_cols, cur_column_graph.nodes, nxt_cols, messages_concat], axis=1),
                                                    globals = tf.concat([prev_globals, cur_column_graph.globals, nxt_globals], axis=1))

        #concatenate with the initial encoding for stability
        #cur_column_graph = gn.utils_tf.concat([cur_column_graph, initial_column_graph], axis = 1)
        cur_column_graph = cur_column_graph.replace(nodes = cur_column_graph.nodes*memberships)

        #update
        cur_column_graph = self.column_network(cur_column_graph)

        #extract messages to sequences by computing means along columns
        messages_out = cur_column_graph.nodes[:, self.config["col_latent_dim"]:]
        messages_to_seq = self.sequence_messenger(messages_out)
        messages_to_alphabet = self.alphabet_messenger(messages_out)

        #cut the messages from the alphabet graph
        cur_column_graph = cur_column_graph.replace(nodes = cur_column_graph.nodes[:, :self.config["col_latent_dim"]])

        return cur_column_graph, messages_to_alphabet, messages_to_seq


    def decode(self, cur_column_graph):
        output_graph = self.column_out_transform(self.column_decoder(cur_column_graph))
        dist_logits = tf.transpose(tf.reshape(output_graph.nodes, [-1, cur_column_graph.n_node[0]]))
        predicted_distribution = tf.nn.softmax(dist_logits)
        return output_graph.nodes, predicted_distribution



#
# A module that learns a representation for each sequence position and updates them
# according to forward edges along the sequences.
# Also maintains a hidden representation of each sequence as a whole that is updated along with
# each individual sequence position.
#
# Each sequence position is initialized with a parameter vector of size T that is tiled accordingly
# where T can be set in Config.py
#
# inputs:   cur_seq_graph - current latent state of the sequences
#           messages_to_seq - a tensor with a message (coming from other modules) for each sequence position
#
# output:   - updates sequences
#           - a message from each sequence position after updating
#
class SequenceKernel(snt.Module):
    def __init__(self, config, name = "SequenceKernel"):

        super(SequenceKernel, self).__init__(name=name)

        self.config = config
        self.lenA = get_len_alphabet(config)

        self.position_param = tf.Variable(
                                    tf.zeros([1, config["seq_latent_dim"]]),
                                        trainable=True, name="position_param")

        self.sequence_param = tf.Variable(
                                    tf.zeros([1, config["seq_latent_dim"]]),
                                        trainable=True, name="sequence_param")

        self.edge_param = tf.Variable(
                                    tf.zeros([1, config["seq_latent_dim"]]),
                                        trainable=True, name="edge_param")

        self.seq_network = gn.modules.GraphNetwork(
                                    edge_model_fn=make_mlp_model(config["seq_net_edge_layers"]+[config["seq_latent_dim"]]),
                                    node_model_fn=make_mlp_model(config["seq_net_node_layers"]+[2*config["seq_latent_dim"]]),
                                    global_model_fn=make_mlp_model(config["seq_net_global_layers"]+[config["seq_latent_dim"]]),
                                    node_block_opt={"use_sent_edges" : True},
                                    reducer = tf.math.unsorted_segment_sum)

        self.alphabet_messenger = make_mlp_model(config["alphabet_to_column_layers"]+[config["alphabet_latent_dim"]])()
        self.column_messenger = make_mlp_model(config["alphabet_to_sequence_layers"]+[config["col_latent_dim"]])()


    #given a topology init_seq for the sequences with forward edges, returns
    #a fully parameterized graph with the same topology (same forward edges)
    def parameterize_init_seq(self, init_seq):
        return init_seq.replace(nodes = tf.tile(self.position_param, [tf.reduce_sum(init_seq.n_node),1]),
                                edges = tf.tile(self.edge_param, [tf.reduce_sum(init_seq.n_edge),1]),
                                globals = tf.tile(self.sequence_param, [gn.utils_tf.get_num_graphs(init_seq),1]))


    def __call__(self, initial_seq_graph, cur_seq_graph, messages_from_alphabet, messages_from_columns, seq_indices, num_col, num_pos):

        #tile alphabet messages and reduce column messages
        messages_from_alphabet = tf.matmul(tf.one_hot(seq_indices, self.lenA), messages_from_alphabet)
        segments = tf.tile(tf.range(num_pos), [num_col])
        messages_from_columns = tf.math.unsorted_segment_sum(messages_from_columns, segments, num_pos)

        #prepare inputs
        cur_seq_graph = cur_seq_graph.replace(nodes = tf.concat([cur_seq_graph.nodes, messages_from_alphabet, messages_from_columns], axis=1))
        #cur_seq_graph = gn.utils_tf.concat([cur_seq_graph, initial_seq_graph], axis = 1)

        #update
        cur_seq_graph = self.seq_network(cur_seq_graph)

        messages_out = cur_seq_graph.nodes[:, self.config["seq_latent_dim"]:]
        messages_to_alphabet = self.alphabet_messenger(messages_out)
        messages_to_columns = self.column_messenger(messages_out)

        cur_seq_graph = cur_seq_graph.replace(nodes = cur_seq_graph.nodes[:, :self.config["seq_latent_dim"]])

        return cur_seq_graph, messages_to_alphabet, messages_to_columns


#
# The model wraps an alphabet kernel, a sequence kernel
# and one or more column kernels (for staged distribution) and handles message passing between them.
#
# Input: init_seq - a graph indicating the topology of the sequences (number of seq, number of nodes, forward edges) and no attributes but
#                   the index of the respective symbol of the alphabet (as integer, not one hot!) per sequence position
#        col_prior - contains a prior probability of membership for each position and column, different columns are required to have different priors
#        col_priors - number of message passing iterations to perform, if multiple column kernels are set, this number applies for each of them,
#                         i.e. the actual number of iterations is #kernels * #iterations
#
# Output: n x R matrix where each line is a probability distribution D_i : P(i in r) for r=1:R
#
class NeuroAlignModel(snt.Module):

    def __init__(self, config, name = "NeuroAlignModel"):

        super(NeuroAlignModel, self).__init__(name=name)
        self.config = config
        self.alphabet_kernel = AlphabetKernel(config)
        self.sequence_kernel = SequenceKernel(config)
        self.column_kernels = [ColumnKernel(config) for _ in range(config["num_col_kernel"] or 1)]
        self.initial_message_seq_2_alpha = tf.Variable(
                                    tf.zeros([1, config["alphabet_latent_dim"]]),
                                        trainable=True, name="initial_message_seq_2_alpha")
        self.initial_message_col_2_alpha = tf.Variable(
                                    tf.zeros([1, config["alphabet_latent_dim"]]),
                                        trainable=True, name="initial_message_col_2_alpha")
        self.initial_message_col_2_seq = tf.Variable(
                                    tf.zeros([1, config["seq_latent_dim"]]),
                                        trainable=True, name="initial_message_col_2_seq")


    def __call__(self, init_seq, col_priors, col_iterations, alpha_iterations, seq_iterations):

        n_pos = col_priors.n_node[0]
        n_col = gn.utils_tf.get_num_graphs(col_priors)

        seq_indices = init_seq.nodes
        sequence_graph = self.sequence_kernel.parameterize_init_seq(init_seq)
        init_sequence_graph = sequence_graph
        alphabet_graph = self.alphabet_kernel.alphabet_param_graph
        message_seq_2_alpha = tf.tile(self.initial_message_seq_2_alpha, [n_pos, 1])
        message_col_2_alpha = tf.tile(self.initial_message_col_2_alpha, [n_pos*n_col, 1])
        message_col_2_seq = tf.tile(self.initial_message_col_2_seq, [n_pos*n_col, 1])
        column_graph = self.column_kernels[0].encode(col_priors)
        memberships = col_priors.nodes

        for column_kernel in self.column_kernels:

            init_column_graph = column_graph

            for _ in range(col_iterations):
                for __ in range(alpha_iterations):
                    alphabet_graph, message_alpha_2_seq, message_alpha_2_col = self.alphabet_kernel(
                                                    alphabet_graph, message_seq_2_alpha, message_col_2_alpha,
                                                    seq_indices, gn.utils_tf.get_num_graphs(col_priors), n_pos)

                for __ in range(seq_iterations):
                    sequence_graph, message_seq_2_alpha, message_seq_2_col = self.sequence_kernel(
                                                    init_sequence_graph, sequence_graph, message_alpha_2_seq, message_col_2_seq,
                                                    seq_indices, gn.utils_tf.get_num_graphs(col_priors), n_pos)

                column_graph, message_col_2_alpha, message_col_2_seq = column_kernel(memberships, init_column_graph, column_graph, message_alpha_2_col, message_seq_2_col, seq_indices)

                memberships,reshaped_memberships = column_kernel.decode(column_graph)

        return reshaped_memberships





class NeuroAlignPredictor():

    def __init__(self, config, examle_msa):

        self.config = config
        self.model = NeuroAlignModel(config)
        self.checkpoint = tf.train.Checkpoint(module=self.model)
        self.checkpoint_root = "./checkpoints"
        self.checkpoint_name = "NeuroAlign"
        self.save_prefix = os.path.join(self.checkpoint_root, self.checkpoint_name)

        def inference(init_seq, col_priors):
            out = self.model(init_seq, col_priors, config["test_col_iterations"], config["test_alpha_iterations_per_col"], config["test_seq_iterations_per_col"])
            return out

        example_seq_g, example_col_g, example_mem = self.get_window_sample(examle_msa, 0, 1, 1)

        # Get the input signature for that function by obtaining the specs
        self.input_signature = [
          gn.utils_tf.specs_from_graphs_tuple(example_seq_g, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(example_col_g, dynamic_num_graphs=True)
        ]

        # Compile the update function using the input signature for speedy code.
        self.inference = tf.function(inference, input_signature=self.input_signature)

    #col_priors is a list of position pairs (s,i) = sequence s at index i
    def predict(self, msa):
        seq_g, col_g, _ = self.get_window_sample(msa, 0, msa.alignment_len-1, msa.alignment_len)#self.config["num_col"])
        mem = self.inference(seq_g, col_g)
        return mem.numpy()


    #constucts sequence graphs and col priors from a msa instance
    #that can be forwarded as input to the predictor
    #Requires an lower- as well as upper-bound for the requested window
    #lb = 0 and ub = alignment_len -1 returns a sample for the complete alignment
    def get_window_sample(self, msa, lb, ub, num_col):

        # retrieves a subset of the sequence positions by windowing
        # and computes targets according to that subset,
        # such that alignment column lb is column 0 in the new sample

        nodes_subset = []
        mem = []
        for seqid, seq in enumerate(msa.raw_seq):
            l = msa.col_to_seq[seqid, lb]
            r = msa.col_to_seq[seqid, ub]
            if r-l>0 or not msa.ref_seq[seqid, lb] == len(msa.alphabet):
                if msa.ref_seq[seqid, lb] == len(msa.alphabet):
                    l += 1
                nodes_subset.append(np.copy(seq[l:(r+1)]))
                lsum = sum(msa.seq_lens[:seqid])
                mem.append(msa.membership_targets[(lsum+l):(lsum+r+1)])

        mem = np.concatenate(mem, axis=0) - lb

        seq_dicts = [{"nodes" : nodes,
                        "senders" : list(range(0, nodes.shape[0]-1)),
                        "receivers" : list(range(1, nodes.shape[0])) }
                        for seqid, nodes in enumerate(nodes_subset)]

        col_dicts = self.make_window_uniform_priors(nodes_subset, num_col)

        def to_graph(dicts):
            g = gn.utils_tf.data_dicts_to_graphs_tuple(dicts)
            g = gn.utils_tf.set_zero_edge_features(g, 0)
            return gn.utils_tf.set_zero_global_features(g, 0)

        return to_graph(seq_dicts), to_graph(col_dicts), mem


    #generates priors for the column graphs such that
    #for each column j, all sequence positions in [c-r, c+r] with c = floor( j*l/L ) have
    #prior probability 1 and other positions 0
    def make_window_uniform_priors(self, nodes, num_col):
        r = self.config["window_uniform_radius"]
        col_prior_dicts = []
        for j in range(1,num_col+1):
            col_nodes = [np.zeros((n.shape[0],1), dtype=np.float32) for n in nodes]
            for cn in col_nodes:
                c = np.floor(j*cn.shape[0]/num_col)
                left = int(max(0, c-r))
                right = int(min(cn.shape[0], c+r+1))
                cn[left:right,:] = 1/(right-left) #uniform probability over sequences
            col_nodes = np.concatenate(col_nodes, axis = 0)
            col_prior_dicts.append({ "nodes" : col_nodes , "senders" : [], "receivers" : []})
        return col_prior_dicts


    def load_latest(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_root)
        if latest is not None:
            self.checkpoint.restore(latest)
            print("Loaded latest checkpoint")


    def save(self):
        self.checkpoint.save(self.save_prefix)
        print("Saved current model.")





#
# #
# class NeuroAlignCore(snt.Module):
#
#     def __init__(self, config, name = "NeuroAlignCore"):
#
#         super(NeuroAlignCore, self).__init__(name=name)
#
#         self.column_network = gn.modules.DeepSets(
#                                     node_model_fn=make_mlp_model(config["column_net_node_layers"]),
#                                     global_model_fn=make_mlp_model(config["column_net_global_layers"]),
#                                     reducer = tf.math.unsorted_segment_mean)
#
#         self.seq_network_en = gn.modules.GraphNetwork(
#                                     edge_model_fn=make_mlp_model(config["seq_net_edge_layers"]),
#                                     node_model_fn=make_mlp_model(config["seq_net_node_layers"]),
#                                     global_model_fn=make_mlp_model(config["seq_net_global_layers"]),
#                                     reducer = tf.math.unsorted_segment_sum)
#
#         self.consensus_seq_network = gn.modules.GraphNetwork(
#                                     edge_model_fn=make_mlp_model(config["consensus_seq_net_edge_layers"]),
#                                     node_model_fn=make_mlp_model(config["consensus_seq_net_node_layers"]),
#                                     global_model_fn=make_mlp_model(config["consensus_seq_net_global_layers"]),
#                                     reducer = tf.math.unsorted_segment_sum)
#
#     def __call__(self, latent_seq_graph, latent_mem, prior_dist, latent_consensus_seq):
#
#         #intra sequence update
#         segments = tf.tile(tf.range(latent_mem.n_node[0]), [gn.utils_tf.get_num_graphs(latent_mem)])
#         reduced_mem = tf.math.unsorted_segment_mean(latent_mem.nodes, segments, latent_mem.n_node[0])
#         seq_in = latent_seq_graph.replace(nodes = tf.concat([latent_seq_graph.nodes, reduced_mem], axis=1))
#         latent_seq_graph = self.seq_network_en(seq_in)
#
#         #consensus sequence update
#         consensus_seq_in = latent_consensus_seq.replace(nodes = tf.concat([latent_consensus_seq.nodes, latent_mem.globals], axis=1))
#         latent_consensus_seq = self.consensus_seq_network(consensus_seq_in)
#
#         #update columns based on their current latent state and the weighted contributing sequence positions
#         nodes_subsets = tf.tile(latent_seq_graph.nodes, [gn.utils_tf.get_num_graphs(latent_mem),1]) #* prior_dist.nodes
#         col_in = latent_mem.replace(nodes = tf.concat([latent_mem.nodes, nodes_subsets], axis=1),
#                                     globals = tf.concat([latent_mem.globals, latent_consensus_seq.nodes], axis=1))
#         latent_mem = self.column_network(col_in)
#
#         return latent_seq_graph, latent_mem, latent_consensus_seq
#
#
#
# #decodes the output of a NeuroAlign core module
# class NeuroAlignDecoder(snt.Module):
#
#     def __init__(self, config, name = "NeuroAlignDecoder"):
#
#         super(NeuroAlignDecoder, self).__init__(name=name)
#
#         self.seq_decoder = gn.modules.GraphIndependent(node_model_fn=make_mlp_model(config["seq_dec_node_layer_s"]))
#
#         self.seq_output_transform = gn.modules.GraphIndependent(node_model_fn = lambda: snt.Linear(1, name="seq_output"))
#
#         self.mem_decoder = gn.modules.GraphIndependent(node_model_fn=make_mlp_model(config["mem_dec_node_layer_s"]),
#                                 global_model_fn=make_mlp_model(config["mem_dec_global_layer_s"]))
#
#         len_alphabet = 4 if config["type"] == "nucleotide" else 23
#         self.mem_output_transform = gn.modules.GraphIndependent(node_model_fn = lambda: snt.Linear(1, name="mem_node_output"),
#                                         global_model_fn = lambda: snt.Linear(1 + len_alphabet + 1, name="mem_global_output"))
#
#
#     def __call__(self, latent_seq_graph, latent_mem, seq_lens):
#
#         #decode and convert to 1d outputs
#         #tf.print(latent_mem.nodes, summarize=-1)
#         seq_out = self.seq_output_transform(self.seq_decoder(latent_seq_graph))
#         mem_out = self.mem_output_transform(self.mem_decoder(latent_mem))
#
#         #get predicted relative positions of sequence nodes and columns
#         node_relative_positions = seq_out.nodes
#         col_relative_positions = mem_out.globals[:,0:1]
#
#         #get (logits of) the predicted relative occurences for each symbol in the alphabet plus the gap symbol
#         rel_occ = mem_out.globals[:,1:]
#         #predict (logits of) the distribution of the seq nodes to the columns using
#         mem_per_col = tf.transpose(tf.reshape(mem_out.nodes, [-1, latent_mem.n_node[0]])) # (num_node, num_pattern)-tensor
#
#         #mpc_dist = tf.sigmoid(mem_per_col)
#
#         exp_mem_per_col = tf.exp(mem_per_col)
#         mpc_dist = tf.sqrt(n_softmax_mem_per_col(exp_mem_per_col)*c_softmax_mem_per_col(exp_mem_per_col, seq_lens))
#         #mpc_dist = tf.nn.softmax(mem_per_col)
#
#         #mpc_dist = mem_per_col
#
#         return node_relative_positions, col_relative_positions, rel_occ, mpc_dist
#
#
# def n_softmax_mem_per_col(exp_mem_per_col):
#     sum_along_cols = tf.reduce_sum(exp_mem_per_col, axis=1, keepdims=True)+1 #+1 to account for softmax uncertainty
#     s_nodes = exp_mem_per_col/sum_along_cols
#     return s_nodes
#
#
# def c_softmax_mem_per_col(exp_mem_per_col, seq_lens):
#     sum_along_seqs = tf.math.segment_sum(exp_mem_per_col, make_seq_ids(seq_lens))+1 #-> (num_segments, num_pattern)-tensor
#     s_cols = exp_mem_per_col/tf.repeat(sum_along_seqs, repeats = seq_lens, axis = 0)
#     return s_cols
#
#
#
#
# #a module that computes the NeuroAlign prediction
# #output is a graph with 1D node and 1D edge attributes
# #nodes are sequence and pattern nodes with respective relative positions
# #edges correspond to region to pattern memberships with logits from which the
# #degrees of membership can be computed
# class NeuroAlignModel(snt.Module):
#
#     def __init__(self, config, name = "NeuroAlignModel"):
#
#         super(NeuroAlignModel, self).__init__(name=name)
#         self.enc = NeuroAlignEncoder(config)
#         self.cores = [NeuroAlignCore(config) for _ in range(config["num_nr_core"])]
#         #self.core = NeuroAlignCore(config)
#         self.dec = NeuroAlignDecoder(config)
#
#
#     # def __call__(self,
#     #             sequence_graph, #input graph with position nodes and forward edges describing the sequences
#     #             seq_lens, #specifies the length of each input sequence, i.e. seq_lens[i] is the length of the ith chain in sequence_graph
#     #             col_priors, #a graph tuple with a col prior graph for each column
#     #             latent_consensus_seq,
#     #             num_iterations):
#     #
#     #     latent_seq_graph, latent_mem = self.enc(sequence_graph, col_priors)
#     #     decoded_outputs = [(None, None, None, tf.transpose(tf.reshape(col_priors.nodes, [-1, col_priors.n_node[0]])))]
#     #     for core in self.cores:
#     #         for _ in range(num_iterations):
#     #             latent_seq_graph, latent_mem, latent_consensus_seq = core(latent_seq_graph, latent_mem, latent_consensus_seq, decoded_outputs[-1][3])
#     #             decoded_outputs.append(self.dec(latent_seq_graph, latent_mem, seq_lens))
#     #     return decoded_outputs[1:]
#
#
#
#     def __call__(self,
#                 sequence_graph, #input graph with position nodes and forward edges describing the sequences
#                 seq_lens, #specifies the length of each input sequence, i.e. seq_lens[i] is the length of the ith chain in sequence_graph
#                 col_priors, #a graph tuple with a col prior graph for each column
#                 subset_g,
#                 latent_consensus_seq,
#                 num_iterations):
#
#         latent_seq_graph, latent_mem = self.enc(sequence_graph, col_priors)
#         decoded_outputs = []
#         for core in self.cores:
#             for _ in range(num_iterations):
#                 latent_seq_graph, latent_mem, latent_consensus_seq = core(latent_seq_graph, latent_mem, col_priors, latent_consensus_seq)
#                 decoded_outputs.append(self.dec(latent_seq_graph, latent_mem, seq_lens))
#         return decoded_outputs
