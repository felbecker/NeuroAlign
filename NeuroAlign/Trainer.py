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
                out = self.predictor.model(sequence_graph, col_priors, config["train_mp_iterations"])
                train_loss = 0
                for n_rp, c_rp, rel_occ, mem_logits in out:
                    l_node_rp = tf.compat.v1.losses.mean_squared_error(target_node_rp, n_rp)
                    l_col_rp = tf.compat.v1.losses.mean_squared_error(col_priors.globals, c_rp)
                    l_rel_occ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = rel_occ_per_col, logits = rel_occ))

                    mem_logits_exp = tf.exp(mem_logits)
                    s_n = Model.n_softmax_mem_per_col(mem_logits_exp)
                    s_c = Model.c_softmax_mem_per_col(mem_logits_exp, len_seqs)
                    # # score_s_n = tf.math.unsorted_segment_prod(s_n, target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors))
                    # # score_s_c = tf.math.unsorted_segment_prod(s_c, target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors))
                    # # score = tf.reduce_mean(score_s_n) + tf.reduce_mean(score_s_c)
                    # score = tf.reduce_mean(tf.math.unsorted_segment_prod(tf.sqrt(s_n*s_c), target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors)))

                    #l_mem_logs = tf.compat.v1.losses.sigmoid_cross_entropy(tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors)), mem_logits)
                    l_mem_logs = tf.compat.v1.losses.log_loss(tf.one_hot(target_col_segment_ids, gn.utils_tf.get_num_graphs(col_priors)), tf.sqrt(s_n*s_c))
                    train_loss += l_node_rp + l_col_rp + l_rel_occ + l_mem_logs
                train_loss /= config["num_nr_core"]*config["train_mp_iterations"]
                regularizer = snt.regularizers.L2(config["l2_regularization"])
                train_loss += regularizer(self.predictor.model.trainable_variables)
                gradients = tape.gradient(train_loss, self.predictor.model.trainable_variables)
                optimizer.apply(gradients, self.predictor.model.trainable_variables)
                return n_rp, c_rp, rel_occ, mem_logits, train_loss, l_node_rp, l_col_rp, l_rel_occ, l_mem_logs

        # Get the input signature for that function by obtaining the specs
        self.input_signature = [
            tf.TensorSpec((None, 1), dtype=tf.dtypes.float32),
            tf.TensorSpec((None), dtype=tf.dtypes.int32),
            tf.TensorSpec((None, config["len_alphabet"]+1), dtype=tf.dtypes.float32)
        ]

        # Compile the update function using the input signature for speedy code.
        self.step_op = tf.function(train_step, input_signature=predictor.input_signature + self.input_signature)

    #use a given reference alignment for training
    def train(self, msa):
        seq_g, col_g = self.predictor._graphs_from_instance(msa, msa.alignment_len)
        return self.step_op(seq_g, col_g, tf.constant(msa.seq_lens), msa.node_rp_targets, msa.membership_targets, msa.rel_occ_per_column)
