import tensorflow as tf
import graph_nets as gn
import sonnet as snt

import Model


#optimizes the NeuroAlign model using a dataset of reference alignments
class NeuroAlignTrainer():

    def __init__(self, config, predictor):
        self.predictor = predictor
        optimizer = snt.optimizers.Adam(config["learning_rate"])
        def train_step(sequence_graph, col_priors, len_seqs,
                        target_node_rp, target_col_segment_ids, rel_occ_per_col):
            with tf.GradientTape() as tape:
                out = self.predictor.model(sequence_graph, len_seqs, col_priors, config["train_mp_iterations"])
                train_loss = 0
                for n_rp, c_rp, rel_occ, mem in out[-1:]:
                    l_node_rp = tf.compat.v1.losses.mean_squared_error(target_node_rp, n_rp)
                    l_col_rp = tf.compat.v1.losses.mean_squared_error(col_priors.globals, c_rp)
                    l_rel_occ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = rel_occ_per_col, logits = rel_occ))
                    #l_mem_logs = tf.compat.v1.losses.sigmoid_cross_entropy(tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors)), mem_logits)
                    l_mem_logs = tf.compat.v1.losses.log_loss(tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors)), mem)
                    l_node_rp = l_node_rp*config["lambda_node_rp"]
                    l_col_rp = l_col_rp*config["lambda_col_rp"]
                    l_rel_occ = l_rel_occ*config["lambda_rel_occ"]
                    l_mem_logs = l_mem_logs*config["lambda_mem"]
                    train_loss += l_node_rp + l_col_rp + l_rel_occ + l_mem_logs
                #train_loss /= config["num_nr_core"]*config["train_mp_iterations"]
                regularizer = snt.regularizers.L2(config["l2_regularization"])
                train_loss += regularizer(self.predictor.model.trainable_variables)
                gradients = tape.gradient(train_loss, self.predictor.model.trainable_variables)
                optimizer.apply(gradients, self.predictor.model.trainable_variables)
                return n_rp, c_rp, rel_occ, mem, train_loss, l_node_rp, l_col_rp, l_rel_occ, l_mem_logs

        # Get the input signature for that function by obtaining the specs
        len_alphabet = 4 if config["type"] == "nucleotide" else 23
        self.input_signature = [
            tf.TensorSpec((None, 1), dtype=tf.dtypes.float32),
            tf.TensorSpec((None), dtype=tf.dtypes.int32),
            tf.TensorSpec((None, len_alphabet+1), dtype=tf.dtypes.float32)
        ]

        # Compile the update function using the input signature for speedy code.
        self.step_op = tf.function(train_step, input_signature=predictor.input_signature + self.input_signature)

    #use a given reference alignment for training
    def train(self, msa):
        seq_g, col_g = self.predictor._graphs_from_instance(msa, msa.alignment_len)
        return self.step_op(seq_g, col_g, tf.constant(msa.seq_lens), msa.node_rp_targets, msa.membership_targets, msa.rel_occ_per_column)