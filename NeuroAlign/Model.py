import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import numpy as np
import os


#all involved neural networks are MLPs with layer sizes defined in Config
#each layer is relu activated
#layer normalization is applied after the last layer
def make_mlp_model(layersizes):
    return lambda: snt.Sequential([
        snt.nets.MLP(layersizes, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])


#converts a 1D int tensor of lenghts to a vector of ids required for segment_mean calls
def make_seq_ids(lens):
    c = tf.cumsum(lens)
    idx = c[:-1]
    s = tf.scatter_nd(tf.expand_dims(idx, 1), tf.ones_like(idx), [c[-1]])
    return tf.cumsum(s)


class DeepSet(gn._base.AbstractModule):

  def __init__(self,
               node_model_fn,
               global_model_fn,
               reducer=tf.math.unsorted_segment_sum,
               name="deep_set"):
    super(DeepSet, self).__init__(name=name)

    with self._enter_variable_scope():
        self._node_block = gn.blocks.NodeBlock(
            node_model_fn=node_model_fn,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=True)
        self._global_block = gn.blocks.GlobalBlock(
            global_model_fn=global_model_fn,
            use_edges=False,
            use_nodes=True,
            use_globals=True,
            nodes_reducer=reducer)

  def _build(self, graph):
    return self._global_block(self._node_block(graph))



#encodes the inputs of the neuroalign model
#after encoding, one or more NeuroAlign Cores follow that them the
#burden of optimization
class NeuroAlignEncoder(snt.Module):

    def __init__(self, config, name = "NeuroAlignEncoder"):

        super(NeuroAlignEncoder, self).__init__(name=name)

        self.seq_encoder = gn.modules.GraphIndependent(
                            edge_model_fn=make_mlp_model(config["seq_enc_edge_layer_s"]),
                            node_model_fn=make_mlp_model(config["seq_enc_node_layer_s"]),
                            global_model_fn=make_mlp_model(config["seq_enc_globals_layer_s"]))

        self.mem_encoder = gn.modules.GraphIndependent(
                            node_model_fn=make_mlp_model(config["mem_enc_node_layer_s"]),
                            global_model_fn=make_mlp_model(config["mem_enc_globals_layer_s"]))

    def __call__(self,
                sequence_graph, #input graph with position nodes and forward edges describing the sequences
                col_priors): #a graph tuple with a col prior graph for each column

        latent_seq_graph = self.seq_encoder(sequence_graph)
        latent_mem = self.mem_encoder(col_priors)
        return latent_seq_graph, latent_mem


#core module that computes the NeuroAlign prediction
#output has to be passed to the NeuroAlignDecoder
class NeuroAlignCore(snt.Module):

    def __init__(self, config, name = "NeuroAlignCore"):

        super(NeuroAlignCore, self).__init__(name=name)

        self.column_network = DeepSet(
                                    node_model_fn=make_mlp_model(config["column_net_node_layers"]),
                                    global_model_fn=make_mlp_model(config["column_net_global_layers"]),
                                    reducer = tf.math.unsorted_segment_mean)

        self.seq_network_en = gn.modules.GraphNetwork(
                                    edge_model_fn=make_mlp_model(config["seq_net_edge_layers"]),
                                    node_model_fn=make_mlp_model(config["seq_net_node_layers"]),
                                    global_model_fn=make_mlp_model(config["seq_net_global_layers"]),
                                    reducer = tf.math.unsorted_segment_sum)

        self.consensus_seq_network = gn.modules.GraphNetwork(
                                    edge_model_fn=make_mlp_model(config["consensus_seq_net_edge_layers"]),
                                    node_model_fn=make_mlp_model(config["consensus_seq_net_node_layers"]),
                                    global_model_fn=make_mlp_model(config["consensus_seq_net_global_layers"]),
                                    reducer = tf.math.unsorted_segment_sum)

    def __call__(self, latent_seq_graph, latent_mem, prior_dist, latent_consensus_seq):

        #intra sequence update
        segments = tf.tile(tf.range(latent_mem.n_node[0]), [gn.utils_tf.get_num_graphs(latent_mem)])
        reduced_mem = tf.math.unsorted_segment_mean(latent_mem.nodes, segments, latent_mem.n_node[0])
        seq_in = latent_seq_graph.replace(nodes = tf.concat([latent_seq_graph.nodes, reduced_mem], axis=1))
        latent_seq_graph = self.seq_network_en(seq_in)

        #consensus sequence update
        consensus_seq_in = latent_consensus_seq.replace(nodes = tf.concat([latent_consensus_seq.nodes, latent_mem.globals], axis=1))
        latent_consensus_seq = self.consensus_seq_network(consensus_seq_in)

        #update columns based on their current latent state and the weighted contributing sequence positions
        nodes_subsets = tf.tile(latent_seq_graph.nodes, [gn.utils_tf.get_num_graphs(latent_mem),1]) #* prior_dist.nodes
        col_in = latent_mem.replace(nodes = tf.concat([latent_mem.nodes, nodes_subsets], axis=1),
                                    globals = tf.concat([latent_mem.globals, latent_consensus_seq.nodes], axis=1))
        latent_mem = self.column_network(col_in)

        return latent_seq_graph, latent_mem, latent_consensus_seq



#decodes the output of a NeuroAlign core module
class NeuroAlignDecoder(snt.Module):

    def __init__(self, config, name = "NeuroAlignDecoder"):

        super(NeuroAlignDecoder, self).__init__(name=name)

        self.seq_decoder = gn.modules.GraphIndependent(node_model_fn=make_mlp_model(config["seq_dec_node_layer_s"]))

        self.seq_output_transform = gn.modules.GraphIndependent(node_model_fn = lambda: snt.Linear(1, name="seq_output"))

        self.mem_decoder = gn.modules.GraphIndependent(node_model_fn=make_mlp_model(config["mem_dec_node_layer_s"]),
                                global_model_fn=make_mlp_model(config["mem_dec_global_layer_s"]))

        len_alphabet = 4 if config["type"] == "nucleotide" else 23
        self.mem_output_transform = gn.modules.GraphIndependent(node_model_fn = lambda: snt.Linear(1, name="mem_node_output"),
                                        global_model_fn = lambda: snt.Linear(1 + len_alphabet + 1, name="mem_global_output"))


    def __call__(self, latent_seq_graph, latent_mem, seq_lens):

        #decode and convert to 1d outputs
        #tf.print(latent_mem.nodes, summarize=-1)
        seq_out = self.seq_output_transform(self.seq_decoder(latent_seq_graph))
        mem_out = self.mem_output_transform(self.mem_decoder(latent_mem))

        #get predicted relative positions of sequence nodes and columns
        node_relative_positions = seq_out.nodes
        col_relative_positions = mem_out.globals[:,0:1]

        #get (logits of) the predicted relative occurences for each symbol in the alphabet plus the gap symbol
        rel_occ = mem_out.globals[:,1:]
        #predict (logits of) the distribution of the seq nodes to the columns using
        mem_per_col = tf.transpose(tf.reshape(mem_out.nodes, [-1, latent_mem.n_node[0]])) # (num_node, num_pattern)-tensor

        #mpc_dist = tf.sigmoid(mem_per_col)

        exp_mem_per_col = tf.exp(mem_per_col)
        mpc_dist = tf.sqrt(n_softmax_mem_per_col(exp_mem_per_col)*c_softmax_mem_per_col(exp_mem_per_col, seq_lens))
        #mpc_dist = tf.nn.softmax(mem_per_col)

        #mpc_dist = mem_per_col

        return node_relative_positions, col_relative_positions, rel_occ, mpc_dist


def n_softmax_mem_per_col(exp_mem_per_col):
    sum_along_cols = tf.reduce_sum(exp_mem_per_col, axis=1, keepdims=True)+1 #+1 to account for softmax uncertainty
    s_nodes = exp_mem_per_col/sum_along_cols
    return s_nodes


def c_softmax_mem_per_col(exp_mem_per_col, seq_lens):
    sum_along_seqs = tf.math.segment_sum(exp_mem_per_col, make_seq_ids(seq_lens))+1 #-> (num_segments, num_pattern)-tensor
    s_cols = exp_mem_per_col/tf.repeat(sum_along_seqs, repeats = seq_lens, axis = 0)
    return s_cols




#a module that computes the NeuroAlign prediction
#output is a graph with 1D node and 1D edge attributes
#nodes are sequence and pattern nodes with respective relative positions
#edges correspond to region to pattern memberships with logits from which the
#degrees of membership can be computed
class NeuroAlignModel(snt.Module):

    def __init__(self, config, name = "NeuroAlignModel"):

        super(NeuroAlignModel, self).__init__(name=name)
        self.enc = NeuroAlignEncoder(config)
        self.cores = [NeuroAlignCore(config) for _ in range(config["num_nr_core"])]
        #self.core = NeuroAlignCore(config)
        self.dec = NeuroAlignDecoder(config)


    # def __call__(self,
    #             sequence_graph, #input graph with position nodes and forward edges describing the sequences
    #             seq_lens, #specifies the length of each input sequence, i.e. seq_lens[i] is the length of the ith chain in sequence_graph
    #             col_priors, #a graph tuple with a col prior graph for each column
    #             latent_consensus_seq,
    #             num_iterations):
    #
    #     latent_seq_graph, latent_mem = self.enc(sequence_graph, col_priors)
    #     decoded_outputs = [(None, None, None, tf.transpose(tf.reshape(col_priors.nodes, [-1, col_priors.n_node[0]])))]
    #     for core in self.cores:
    #         for _ in range(num_iterations):
    #             latent_seq_graph, latent_mem, latent_consensus_seq = core(latent_seq_graph, latent_mem, latent_consensus_seq, decoded_outputs[-1][3])
    #             decoded_outputs.append(self.dec(latent_seq_graph, latent_mem, seq_lens))
    #     return decoded_outputs[1:]



    def __call__(self,
                sequence_graph, #input graph with position nodes and forward edges describing the sequences
                seq_lens, #specifies the length of each input sequence, i.e. seq_lens[i] is the length of the ith chain in sequence_graph
                col_priors, #a graph tuple with a col prior graph for each column
                subset_g,
                latent_consensus_seq,
                num_iterations):

        latent_seq_graph, latent_mem = self.enc(sequence_graph, col_priors)
        decoded_outputs = []
        for core in self.cores:
            for _ in range(num_iterations):
                latent_seq_graph, latent_mem, latent_consensus_seq = core(latent_seq_graph, latent_mem, col_priors, latent_consensus_seq)
                decoded_outputs.append(self.dec(latent_seq_graph, latent_mem, seq_lens))
        return decoded_outputs

class NeuroAlignPredictor():

    def __init__(self, config, examle_msa):

        self.config = config
        self.model = NeuroAlignModel(config)
        self.checkpoint = tf.train.Checkpoint(module=self.model)
        self.checkpoint_root = "./checkpoints"
        self.checkpoint_name = "NeuroAlign"
        self.save_prefix = os.path.join(self.checkpoint_root, self.checkpoint_name)

        def inference(sequence_graph, col_priors, subset_g, inter_seq, len_seqs):
            out = self.model(sequence_graph, len_seqs, col_priors, subset_g, inter_seq, config["test_mp_iterations"])
            node_relative_pos, col_relative_pos, rel_occ, mem_per_col = out[-1]
            return node_relative_pos, col_relative_pos, rel_occ, mem_per_col

        seq_input_example, col_input_example, subset_input_example, consensus_seq_input_example,_,_,_ = self.get_window_sample(examle_msa, 1, 0, 1)

        # Get the input signature for that function by obtaining the specs
        self.input_signature = [
          gn.utils_tf.specs_from_graphs_tuple(seq_input_example, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(col_input_example, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(subset_input_example, dynamic_num_graphs=True),
          gn.utils_tf.specs_from_graphs_tuple(consensus_seq_input_example),
          tf.TensorSpec((None,), dtype=tf.dtypes.int32)
        ]

        # Compile the update function using the input signature for speedy code.
        self.inference = tf.function(inference, input_signature=self.input_signature)

    #col_priors is a list of position pairs (s,i) = sequence s at index i
    def predict(self, msa, num_cols):
        seq_g, col_g, subset_g, consensus_seq_g,_,_,_ = self.get_window_sample(msa, num_cols, 0, msa.alignment_len-1)
        node_relative_pos, col_relative_pos, rel_occ, mem_per_col = self.inference(seq_g, col_g, subset_g, consensus_seq_g, tf.constant(msa.seq_lens))
        return node_relative_pos.numpy(), col_relative_pos.numpy(), tf.nn.softmax(rel_occ).numpy(), mem_per_col.numpy()


    #constucts graphs that can be forwarded as input to the predictor
    #from a msa instance. Requires an approximation of the alignment length
    #and an lower- as well as upper-bound for the requested window
    # [[for training]]
    def get_window_sample(self, msa, alignment_len, lb, ub):

        # construct a subset of the sequences and targets according to that subset,
        # such that alignment column lb is column 0 in the new sample

        nodes_subset = []
        mem = []
        for seqid, nodes in enumerate(msa.nodes):
            l = msa.col_to_seq[seqid, lb]
            r = msa.col_to_seq[seqid, ub]
            if r-l>0 or not msa.ref_seq[seqid, lb] == len(msa.alphabet):
                if msa.ref_seq[seqid, lb] == len(msa.alphabet):
                    l += 1
                nodes_subset.append(np.copy(nodes[l:(r+1), :len(msa.alphabet)]))
                #nodes_subset.append(np.copy(nodes[l:(r+1)]))
                #nodes_subset[-1][:,len(msa.alphabet)] -= (l-1)
                lsum = sum(msa.seq_lens[:seqid])
                mem.append(msa.membership_targets[(lsum+l):(lsum+r+1)])

        mem = np.concatenate(mem, axis=0) - lb

        # for nodes in nodes_subset:
        #     nodes[:,len(msa.alphabet)] /= nodes.shape[0]

        sl = [nodes.shape[0] for nodes in nodes_subset]

        seq_dicts = [{"globals" : [nodes.shape[0] / (ub-lb+1)],  #[np.float32(0)],
                        "nodes" : nodes,
                        "edges" : np.zeros((nodes.shape[0]-1, 1), dtype=np.float32),
                        "senders" : list(range(0, nodes.shape[0]-1)),
                        "receivers" : list(range(1, nodes.shape[0])) }
                        for seqid, nodes in enumerate(nodes_subset)]

        subset_dicts, col_dicts = self.make_window_uniform_priors(nodes_subset, ub-lb+1)

        seq_g = gn.utils_tf.data_dicts_to_graphs_tuple(seq_dicts)
        col_g = gn.utils_tf.data_dicts_to_graphs_tuple(col_dicts)
        col_g = gn.utils_tf.set_zero_edge_features(col_g, 0)
        subset_g = gn.utils_tf.data_dicts_to_graphs_tuple(subset_dicts)
        subset_g = gn.utils_tf.set_zero_edge_features(subset_g, 0)

        rocc = msa.rel_occ_per_column[lb:(ub+1),:]

        # print(lb, ub)
        # print(msa.ref_seq)
        # print(mem)

        consensus_seq_dict = {"nodes" : np.zeros((ub-lb+1 ,self.config["hidden_dim"]), dtype=np.float32),
                          "globals" : np.zeros((self.config["hidden_dim"]), dtype=np.float32),
                          "edges" : np.zeros((ub-lb ,self.config["hidden_dim"]), dtype=np.float32),
                          "senders" : list(range(0, ub-lb)),
                          "receivers" : list(range(1, ub-lb+1))}
        consensus_seq_g = gn.utils_tf.data_dicts_to_graphs_tuple([consensus_seq_dict])

        return seq_g, col_g, subset_g, consensus_seq_g, sl, mem, rocc


    #generates priors for the column graphs such that
    #for each column j, all sequence positions in [c-r, c+r] with c = floor( j*l/L ) are
    #weighted 1/(2r+1) and other positions are weighted 0
    def make_window_uniform_priors(self, nodes, num_col):
        r = self.config["window_uniform_radius"]
        subset_dicts = []
        col_prior_dicts = []
        seq_nodes = np.concatenate(nodes, axis = 0)
        for j in range(1,num_col+1):
            col_nodes = [np.zeros((n.shape[0],1), dtype=np.float32) for n in nodes]
            for cn in col_nodes:
                c = np.floor(j*cn.shape[0]/num_col)-1
                left = int(max(0, c-r))
                right = int(min(cn.shape[0], c+r+1))
                cn[left:right,:] = 1
            col_nodes = np.concatenate(col_nodes, axis = 0)
            subset_dicts.append({"globals" : [np.float32(0)], #j/num_col
            "nodes" : seq_nodes*col_nodes,
            "senders" : [],
            "receivers" : [] })
            col_prior_dicts.append({"globals" : [np.float32(0)], #j/num_col
            "nodes" : col_nodes,
            "senders" : [],
            "receivers" : [] })
        return subset_dicts, col_prior_dicts


    def load_latest(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_root)
        if latest is not None:
            self.checkpoint.restore(latest)
            print("Loaded latest checkpoint")


    def save(self):
        self.checkpoint.save(self.save_prefix)
        print("Saved current model.")
