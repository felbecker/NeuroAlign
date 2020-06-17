import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import numpy as np

import Model


class NeuroAlignTrainer():

    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor
        optimizer = snt.optimizers.Adam(config["learning_rate"])
        def train_step(sequence_graph, col_graph, priors, target_col_ids):
            with tf.GradientTape() as tape:
                memberships = self.predictor.model(sequence_graph, col_graph, priors, config["train_iterations"])
                train_loss = 0
                mem_tar = tf.one_hot(target_col_ids, col_graph.n_node[0])
                mem_tar_sqr = tf.matmul(mem_tar, mem_tar, transpose_b = True)
                for mem in memberships:
                    mem_sqr = tf.matmul(mem, mem, transpose_b = True)
                    l_mem = tf.compat.v1.losses.log_loss(labels=mem_tar_sqr, predictions=mem_sqr)
                    train_loss += l_mem
                train_loss /= config["train_iterations"]
                regularizer = snt.regularizers.L2(config["l2_regularization"])
                train_loss += regularizer(self.predictor.model.trainable_variables)
                gradients = tape.gradient(train_loss, self.predictor.model.trainable_variables)
                optimizer.apply(gradients, self.predictor.model.trainable_variables)
                return mem, train_loss, l_mem

        # Get the input signature for that function by obtaining the specs
        self.input_signature = [
            tf.TensorSpec((None), dtype=tf.dtypes.int32)
        ]

        # Compile the update function using the input signature for speedy code.
        self.step_op = tf.function(train_step, input_signature=predictor.input_signature + self.input_signature)

    #supervised training using reference alignment
    #randomly select a center columns with a window of adjacent columns within
    #a certain radius (if existing)
    def train(self, msa):
        c = np.random.randint(msa.alignment_len)
        lb = max(0, c - self.config["adjacent_column_radius"])
        ub = min(msa.alignment_len-1, c + self.config["adjacent_column_radius"])
        seq_g, col_g, priors, target_col_ids = self.predictor.get_window_sample(msa, lb, ub, ub -lb +1)#self.config["num_col"])
        return self.step_op(seq_g, col_g, priors, target_col_ids)



#optimizes the NeuroAlign model using a dataset of reference alignments
# class NeuroAlignTrainer():
#
#     def __init__(self, config, predictor):
#         self.config = config
#         self.predictor = predictor
#         optimizer = snt.optimizers.Adam(config["learning_rate"])
#         def train_step(sequence_graph, col_priors, subset_g, consensus_seq, len_seqs,
#                         target_node_rp, target_col_segment_ids, rel_occ_per_col):
#             with tf.GradientTape() as tape:
#                 out = self.predictor.model(sequence_graph, len_seqs, col_priors, subset_g, consensus_seq, config["train_mp_iterations"])
#                 train_loss = 0
#                 for n_rp, c_rp, rel_occ, mem in out: #out[-1:]
#                     l_node_rp = tf.compat.v1.losses.mean_squared_error(target_node_rp, n_rp)
#                     l_col_rp = tf.compat.v1.losses.mean_squared_error(col_priors.globals, c_rp)
#                     l_rel_occ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = rel_occ_per_col, logits = rel_occ))
#
#                     #l_mem_logs = tf.compat.v1.losses.sigmoid_cross_entropy(tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors)), mem_logits)
#                     #l_mem_logs = tf.compat.v1.losses.log_loss(tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors)), mem)   #tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors))
#
#                     #col_products = tf.math.unsorted_segment_prod(mem, target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors))
#                     #invers_mem = 1-mem
#                     #col_complement_products = tf.math.reduce_prod(invers_mem, axis=0, keepdims=True) / tf.math.unsorted_segment_prod(invers_mem, target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors))
#                     #l_mem_logs = -tf.reduce_sum(col_products*col_complement_products) / tf.cast(gn.utils_tf.get_num_graphs(col_priors), tf.float32)
#
#                     mem_tar = tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors))
#                     mem_tar = tf.matmul(mem_tar, mem_tar, transpose_b = True)
#                     mem_pred = tf.matmul(mem, mem, transpose_b = True)
#                     l_mem_logs = tf.compat.v1.losses.log_loss(mem_tar, mem_pred)
#
#                     l_node_rp = l_node_rp*config["lambda_node_rp"]
#                     l_col_rp = l_col_rp*config["lambda_col_rp"]
#                     l_rel_occ = l_rel_occ*config["lambda_rel_occ"]
#                     l_mem_logs = l_mem_logs*config["lambda_mem"]
#                     train_loss += l_node_rp + l_col_rp + l_rel_occ + l_mem_logs
#                     #train_loss += l_mem_logs
#                 train_loss /= config["num_nr_core"]*config["train_mp_iterations"]
#                 regularizer = snt.regularizers.L2(config["l2_regularization"])
#                 train_loss += regularizer(self.predictor.model.trainable_variables)
#                 gradients = tape.gradient(train_loss, self.predictor.model.trainable_variables)
#                 optimizer.apply(gradients, self.predictor.model.trainable_variables)
#                 #tf.print(mem, summarize=-1)
#                 return n_rp, c_rp, rel_occ, mem, train_loss, l_node_rp, l_col_rp, l_rel_occ, l_mem_logs
#
#         # Get the input signature for that function by obtaining the specs
#         len_alphabet = 4 if config["type"] == "nucleotide" else 23
#         self.input_signature = [
#             tf.TensorSpec((None, 1), dtype=tf.dtypes.float32),
#             tf.TensorSpec((None), dtype=tf.dtypes.int32),
#             tf.TensorSpec((None, len_alphabet+1), dtype=tf.dtypes.float32)
#         ]
#
#         # Compile the update function using the input signature for speedy code.
#         self.step_op = tf.function(train_step, input_signature=predictor.input_signature + self.input_signature)
#
#     #supervised training using reference alignment
#     #randomly select a center columns with a window of adjacent columns within
#     #a certain radius (if existing)
#     def train(self, msa):
#         c = np.random.randint(msa.alignment_len)
#         lb = max(0, c - self.config["adjacent_column_radius"])
#         ub = min(msa.alignment_len-1, c + self.config["adjacent_column_radius"])
#         seq_g, col_g, subset_g, consensus_seq_g, sl, mem, rocc = self.predictor.get_window_sample(msa, msa.alignment_len, lb, ub)
#         #print("_______________________________________________________")
#         #print(seq_g.nodes, sl, mem, rocc)
#         return self.step_op(seq_g, col_g, subset_g, consensus_seq_g, tf.constant(sl), np.reshape((mem+1)/(ub-lb+1), (-1,1)), mem, rocc)
