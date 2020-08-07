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

class Identity(snt.Module):
    def __call__(self, x):
        return x

def make_identity():
    return lambda: Identity()

def make_lstm_model(layersizes):
    return lambda: snt.DeepRNN([snt.LSTM(s) for s in layersizes])

def get_len_alphabet(config):
    return 4 if config["type"] == "nucleotide" else 23

def init_weights(shape):
    return np.random.normal(0, 0.1, shape).astype(dtype=np.float32)

#
# A graph network that performs update steps according to the graph
# topology as usual, but with LSTM update functions and hidden states
# accompanying edges, nodes and globals
#
class LSTMGraphNetwork(gn._base.AbstractModule):
    def __init__(self,
                edge_lstm,
                node_lstm,
                global_lstm,
                reducer=tf.math.unsorted_segment_sum,
                edge_block_opt=None,
                node_block_opt=None,
                global_block_opt=None,
                name = "LSTMGraphNetwork"):
        super(LSTMGraphNetwork, self).__init__(name=name)
        edge_block_opt = gn.modules._make_default_edge_block_opt(edge_block_opt)
        node_block_opt = gn.modules._make_default_node_block_opt(node_block_opt, reducer)
        global_block_opt = gn.modules._make_default_global_block_opt(global_block_opt, reducer)

        with self._enter_variable_scope():
            self._edge_block = gn.blocks.EdgeBlock(
                edge_model_fn=make_identity(), **edge_block_opt)
            self._node_block = gn.blocks.NodeBlock(
                node_model_fn=make_identity(), **node_block_opt)
            self._global_block = gn.blocks.GlobalBlock(
                global_model_fn=make_identity(), **global_block_opt)
            self._edge_lstm = edge_lstm()
            self._node_lstm = node_lstm()
            self._global_lstm = global_lstm()


    def get_initial_states(self, template_graph):
        init_edges = self._edge_lstm.initial_state(batch_size=tf.reduce_sum(template_graph.n_edge))
        init_nodes = self._node_lstm.initial_state(batch_size=tf.reduce_sum(template_graph.n_node))
        init_global = self._global_lstm.initial_state(batch_size=gn.utils_tf.get_num_graphs(template_graph))
        return template_graph.replace(edges=init_edges, nodes=init_nodes, globals=init_global)



    def _build(self, graph, hidden_graph):

        graph_e = self._edge_block(graph)
        output_edges, hidden_edges = self._edge_lstm(graph_e.edges, hidden_graph.edges)
        graph_e = graph_e.replace(edges=output_edges)

        graph_n = self._node_block(graph_e)
        output_nodes, hidden_nodes = self._node_lstm(graph_n.nodes, hidden_graph.nodes)
        graph_n = graph_n.replace(nodes=output_nodes)

        graph_g = self._global_block(graph_n)
        output_globals, hidden_globals = self._global_lstm(graph_n.globals, hidden_graph.globals)
        graph_g = graph_g.replace(globals=output_globals)

        updated_hidden_graph = hidden_graph.replace(
                                nodes=hidden_nodes,
                                edges=hidden_edges,
                                globals=hidden_globals)

        return graph_g, updated_hidden_graph



#
# A module that learns a representation for each sequence position and updates them
# according to forward edges along the sequences.
# Also maintains a global representation of each sequence that is updated along with
# each individual sequence position and a global representation for all sequences.
#
# Each sequence position is initialized with a shared parameter vector of size T that is tiled accordingly
# where T can be set in Config.py
#
# inputs:   sequence_graph - current state of the sequences
#           alignment_global - current global state of the alignment
#           alphabet_graph - current state of the alphabet
#           column_graph - current state of the columns
#           memberships - matrix of membership degrees
#
# output:   - updated sequences
#
class SequenceKernel(gn._base.AbstractModule):
    def __init__(self, config, name = "SequenceKernel"):

        super(SequenceKernel, self).__init__(name=name)

        self.config = config
        self.lenA = get_len_alphabet(config)

        with self._enter_variable_scope():
            self.sequence_param = tf.Variable(
                                        tf.zeros([1, config["seq_latent_dim"]]),
                                            trainable=True, name="sequence_param")

            self.edge_param = tf.Variable(
                                        tf.zeros([1, config["seq_latent_dim"]]),
                                            trainable=True, name="edge_param")

            self.seq_global_param = tf.Variable(
                                        tf.zeros([1, config["seq_global_dim"]]),
                                            trainable=True, name="seq_global_param")

            self.seq_network = LSTMGraphNetwork(
                                        edge_lstm=make_lstm_model(config["seq_net_edge_layers"]+[config["seq_latent_dim"]]),
                                        node_lstm=make_lstm_model(config["seq_net_node_layers"]+[config["seq_latent_dim"]]),
                                        global_lstm=make_lstm_model(config["seq_net_global_per_seq_layers"]+[config["seq_latent_dim"]]),
                                        node_block_opt={"use_sent_edges" : True},
                                        reducer = tf.math.unsorted_segment_mean)


            self.node_param = tf.Variable(
                                        init_weights([self.lenA, config["seq_latent_dim"]]),
                                            trainable=True, name="alphabet_node_param")

            self.column_messenger = make_mlp_model(config["columns_to_sequence_layers"]+[config["col_latent_dim"]])()

            self.node_encoder = make_mlp_model(config["encoder"]+[config["col_latent_dim"]])()


    #given a topology init_seq for the sequences with forward edges, returns
    #a fully parameterized graph with the same topology
    def parameterize_init_seq(self, init_seq):
        symbol_embeddings = tf.matmul(tf.one_hot(tf.cast(init_seq.nodes[:,0], tf.int32), self.lenA), self.node_param)
        rp = init_seq.nodes[:,1:2]
        nodes = self.node_encoder(tf.concat([symbol_embeddings, rp], axis=1))
        return init_seq.replace(nodes = nodes,
                                edges = tf.tile(self.edge_param, [tf.reduce_sum(init_seq.n_edge),1]),
                                globals = tf.tile(self.sequence_param, [gn.utils_tf.get_num_graphs(init_seq),1])), self.seq_global_param


    def _build(self, init_nodes, sequence_graph, hidden_sequence_graph, column_graph, memberships):

        #compute incoming messages from columns
        messages_from_columns = self.column_messenger(column_graph.nodes)
        messages_from_columns = tf.matmul(memberships, messages_from_columns)

        #concat and update
        in_sequence_graph = sequence_graph.replace(nodes = tf.concat([init_nodes, sequence_graph.nodes, messages_from_columns], axis=1))
        out_sequence_graph, hidden_sequence_graph = self.seq_network(in_sequence_graph, hidden_sequence_graph)

        return out_sequence_graph, hidden_sequence_graph






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
# inputs:   sequence_graph - current state of the sequences
#           column_graph - current state of the columns
#           memberships - matrix of membership degrees
#
# output: updated representation of the columns
#         messages to be send to every sequence node
#
class ColumnKernel(gn._base.AbstractModule):
    def __init__(self, config, name = "ColumnKernel"):

        super(ColumnKernel, self).__init__(name=name)

        self.config = config
        self.lenA = get_len_alphabet(config)

        with self._enter_variable_scope():
            self.column_network = LSTMGraphNetwork(
                                        node_lstm=make_lstm_model(config["column_net_node_layers"]+[config["col_latent_dim"]]),
                                        edge_lstm=make_lstm_model(config["column_net_edge_layers"]+[config["col_latent_dim"]]),
                                        global_lstm=make_lstm_model(config["column_net_global_layers"]+[config["col_latent_dim"]]),
                                        node_block_opt={"use_sent_edges" : True},
                                        reducer= tf.math.unsorted_segment_mean)

            self.col_global_param = tf.Variable(
                                        tf.zeros([1, config["col_latent_dim"]]),
                                            trainable=True, name="col_global_param")

            self.col_edge_param = tf.Variable(
                                        tf.zeros([1, config["col_latent_dim"]]),
                                            trainable=True, name="col_edge_param")

            self.sequence_messenger = make_mlp_model(config["sequence_to_columns_layers"]+[config["seq_latent_dim"]])()

            self.col_encoder = make_mlp_model([config["col_latent_dim"]])()



    def parameterize_col_priors(self, col_graph):
        return col_graph.replace(nodes = self.col_encoder(col_graph.nodes),
                                edges = tf.tile(self.col_edge_param, [col_graph.n_edge[0],1]),
                                globals = self.col_global_param)



    def _build(self, init_nodes, column_graph, hidden_column_graph, sequence_graph, memberships):

        #compute incoming messages from sequences
        messages_from_seq = self.sequence_messenger(sequence_graph.nodes)
        messages_from_seq = tf.matmul(memberships, messages_from_seq, transpose_a = True)

        #update
        in_column_graph = column_graph.replace(nodes = tf.concat([init_nodes, column_graph.nodes, messages_from_seq], axis=1))
        out_column_graph, hidden_column_graph = self.column_network(in_column_graph, hidden_column_graph)

        return out_column_graph, hidden_column_graph



#
# The model updates sequence and column states iteratively by passing messages between sites and columns and between subsequent sites.
#
# Input: init_seq - a graph indicating the topology of the sequences (number of seq, number of nodes, forward edges) and no attributes but
#                   the index of the respective symbol of the alphabet (as integer, not one hot!) per sequence position
#        col_prior - contains a prior probability of membership for each position and column, different columns are required to have different priors
#        iterations - number of message passing iterations to perform
#
# Output: n x R matrix where each line is a probability distribution D_i : P4(i in r) for r=1:R
#
class NeuroAlignModel(gn._base.AbstractModule):

    def __init__(self, config, name = "NeuroAlignModel"):

        super(NeuroAlignModel, self).__init__(name=name)
        self.config = config

        with self._enter_variable_scope():
            self.sequence_kernel = [SequenceKernel(config) for _ in range(self.config["num_kernel"])]
            self.column_kernel = [ColumnKernel(config) for _ in range(self.config["num_kernel"])]

            #self.membership_decoder = snt.DeepRNN([snt.LSTM(s) for s in config["column_decode_node_layers"]])
            self.membership_decoder = make_mlp_model(config["column_decode_node_layers"])()
            self.membership_out_transform = snt.Linear(1, name="column_out_transform")

            self.rp_decoder = make_mlp_model([config["col_latent_dim"]])()
            self.rp_out = snt.Linear(1, name="rp_out_transform")


    # def _build(self, init_seq, init_cols, membership_priors, iterations):
    #
    #     sequence_graph, alignment_global = self.sequence_kernel.parameterize_init_seq(init_seq)
    #     init_seq = sequence_graph.nodes
    #     column_graph = self.column_kernel.parameterize_col_priors(init_cols)
    #     init_col = column_graph.nodes
    #     memberships = [membership_priors]
    #     relative_positions = []
    #     #mem_decode_state = self.membership_decoder.initial_state(batch_size=tf.reduce_sum(sequence_graph.n_node)*column_graph.n_node[0])
    #     hidden_sequence_graph = self.sequence_kernel.seq_network.get_initial_states(sequence_graph)
    #     hidden_column_graph = self.column_kernel.column_network.get_initial_states(column_graph)
    #     for _ in range(iterations):
    #         sequence_graph, hidden_sequence_graph = self.sequence_kernel(init_seq, sequence_graph, hidden_sequence_graph, column_graph, memberships[-1])
    #         column_graph, hidden_column_graph = self.column_kernel(init_col, column_graph, hidden_column_graph, sequence_graph,  memberships[-1])
    #         mem = self.decode(init_seq, init_col, column_graph, sequence_graph)
    #         memberships.append(mem)
    #         relative_positions.append(self.rp_out(self.rp_decoder(tf.concat([init_seq, sequence_graph.nodes], axis=1))))
    #     return memberships[1:], relative_positions


    def _build(self, init_seq, init_cols, membership_priors, iterations):

        sequence_graph, alignment_global = self.sequence_kernel[0].parameterize_init_seq(init_seq)
        init_seq = sequence_graph.nodes
        column_graph = self.column_kernel[0].parameterize_col_priors(init_cols)
        init_col = column_graph.nodes
        memberships = [membership_priors]
        running_mem = membership_priors
        relative_positions = []
        #mem_decode_state = self.membership_decoder.initial_state(batch_size=tf.reduce_sum(sequence_graph.n_node)*column_graph.n_node[0])
        hidden_sequence_graph = self.sequence_kernel[0].seq_network.get_initial_states(sequence_graph)
        hidden_column_graph = self.column_kernel[0].column_network.get_initial_states(column_graph)
        for num_kernel in range(self.config["num_kernel"]):
            for _ in range(iterations):
                sequence_graph, hidden_sequence_graph = self.sequence_kernel[num_kernel](init_seq, sequence_graph, hidden_sequence_graph, column_graph, running_mem)
                column_graph, hidden_column_graph = self.column_kernel[num_kernel](init_col, column_graph, hidden_column_graph, sequence_graph,  running_mem)
                mem = self.decode(init_seq, init_col, column_graph, sequence_graph)
                memberships.append(mem)
                running_mem = 0.1*running_mem + 0.9*memberships[-1]
                relative_positions.append(self.rp_out(self.rp_decoder(tf.concat([init_seq, sequence_graph.nodes], axis=1))))
        return memberships, relative_positions



    #decodes the states of sequences S_i, columns C_r to membership probabilities P(i in r | S_i, C_r)
    def decode(self, init_seq, init_col, column_graph, sequence_graph):
        n_pos = tf.reduce_sum(sequence_graph.n_node)
        n_col = column_graph.n_node[0]
        seq = tf.concat([init_seq, sequence_graph.nodes], axis=1)
        col = tf.concat([init_col, column_graph.nodes], axis=1)
        positions = tf.repeat(seq, tf.repeat(n_col, n_pos), axis=0)
        columns = tf.tile(col, [n_pos, 1])
        decode_in = tf.concat([positions, columns], axis=1)
        latent_out = self.membership_decoder(decode_in)
        decode_out = self.membership_out_transform(latent_out)
        memberships = tf.nn.softmax(tf.reshape(decode_out, [n_pos, n_col]))
        return memberships





class NeuroAlignPredictor():

    def __init__(self, config, examle_msa):

        self.config = config
        self.model = NeuroAlignModel(config)
        self.checkpoint = tf.train.Checkpoint(module=self.model)
        self.checkpoint_root = "./checkpoints"
        self.checkpoint_name = "NeuroAlign"
        self.save_prefix = os.path.join(self.checkpoint_root, self.checkpoint_name)

        def inference(init_seq, init_cols, priors):
            out,rp = self.model(init_seq, init_cols, priors, config["test_iterations"])
            return out[-1], rp[-1]

        example_seq_g, example_col_g, example_priors, example_mem = self.get_window_sample(examle_msa, 0, 1, 1)

        # Get the input signature for that function by obtaining the specs
        self.input_signature = [
          gn.utils_tf.specs_from_graphs_tuple(example_seq_g, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(example_col_g, dynamic_num_graphs=False),
          tf.TensorSpec((None, None), dtype=tf.dtypes.float32)
        ]

        # Compile the update function using the input signature for speedy code.
        self.inference = tf.function(inference, input_signature=self.input_signature)

    #col_priors is a list of position pairs (s,i) = sequence s at index i
    def predict(self, msa):
        seq_g, col_g, priors, _ = self.get_window_sample(msa, 0, msa.alignment_len-1, msa.alignment_len)
        mem, rp = self.inference(seq_g, col_g, priors)
        # out = self.model(init_seq, init_cols, priors, config["test_iterations"])
        # mem = out[-1]
        return mem.numpy(), rp.numpy()


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

        seq_dicts = [{"nodes" : np.concatenate((np.reshape(nodes, (-1,1)),
                                                np.reshape(np.linspace(0,1,nodes.shape[0]), (-1,1))),
                                                    axis=1).astype(np.float32),
                        "senders" : list(range(0, nodes.shape[0]-1)),
                        "receivers" : list(range(1, nodes.shape[0])) }
                        for seqid, nodes in enumerate(nodes_subset)]

        col_dict, priors = self.make_window_uniform_priors(nodes_subset, num_col)

        def to_graph(dicts):
            g = gn.utils_tf.data_dicts_to_graphs_tuple(dicts)
            g = gn.utils_tf.set_zero_edge_features(g, 0)
            return gn.utils_tf.set_zero_global_features(g, 0)

        return to_graph(seq_dicts), to_graph([col_dict]), priors, mem


    #generates priors for the column graphs such that
    #for each column j, all sequence positions in [c-r, c+r] with c = floor( j*l/L ) have
    #prior probability 1 and other positions 0
    def make_window_uniform_priors(self, nodes, num_col):
        r = self.config["window_uniform_radius"]
        seq_lens = [seq.shape[0] for seq in nodes]
        memberships = [np.zeros([l, num_col], dtype=np.float32) for l in seq_lens]
        for cnodes, l in zip(memberships, seq_lens):
            for i in range(l):
                c = np.floor(i*num_col/l)
                left = int(max(0, c-r))
                right = int(min(num_col, c+r+1))
                cnodes[i,left:right] = 1/(right-left)
        memberships =  np.concatenate(memberships, axis = 0)
        col_prior_dict = {"nodes" : np.reshape(np.linspace(0, 1, num_col, dtype=np.float32), [num_col,1]),
                            "senders" : list(range(0, num_col-1)),
                          "receivers" : list(range(1, num_col)) }
        return col_prior_dict, memberships


    def load_latest(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_root)
        if latest is not None:
            self.checkpoint.restore(latest).expect_partial()
            print("Loaded latest checkpoint")


    def save(self):
        self.checkpoint.save(self.save_prefix)
        print("Saved current model.")
