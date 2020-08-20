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
        def train_step(sequence_graph, col_graph, priors, target_col_ids, target_gaps_in, target_gaps_begin, target_gaps_end):
            with tf.GradientTape() as tape:
                memberships, relative_positions, gaps = self.predictor.model(sequence_graph, col_graph, priors, config["train_iterations"])
                train_loss = 0
                mem_tar = tf.one_hot(target_col_ids, col_graph.n_node[0])
                mem_tar_sqr = tf.matmul(mem_tar, mem_tar, transpose_b = True)
                rp_targets = tf.reshape(target_col_ids/col_graph.n_node[0], (-1,1))
                weights = [1]*(len(memberships)-1) + [config["final_iteration_loss_weight"]]
                for mem, rp, (g,gs,ge), w in zip(memberships, relative_positions, gaps, weights):
                    mem_sqr = tf.matmul(mem, mem, transpose_b = True)
                    l_mem = tf.compat.v1.losses.log_loss(labels=mem_tar_sqr, predictions=mem_sqr)
                    l_rp = tf.compat.v1.losses.mean_squared_error(labels=rp_targets, predictions=rp)
                    l_g = tf.compat.v1.losses.mean_squared_error(labels=tf.reshape(target_gaps_in/col_graph.n_node[0], (-1,1)), predictions=g)
                    l_g += tf.compat.v1.losses.mean_squared_error(labels=tf.reshape(target_gaps_begin/col_graph.n_node[0], (-1,1)), predictions=gs)
                    l_g += tf.compat.v1.losses.mean_squared_error(labels=tf.reshape(target_gaps_end/col_graph.n_node[0], (-1,1)), predictions=ge)
                    train_loss += w*(l_mem + config["lambda_rp"]*l_rp + config["lambda_gap"]*l_g)
                train_loss /= sum(weights)
                regularizer = snt.regularizers.L2(config["l2_regularization"])
                train_loss += regularizer(self.predictor.model.trainable_variables)
                gradients = tape.gradient(train_loss, self.predictor.model.trainable_variables)
                optimizer.apply(gradients, self.predictor.model.trainable_variables)
                return mem, train_loss, l_mem, l_rp, l_g

        # Get the input signature for that function by obtaining the specs
        self.input_signature = [
            tf.TensorSpec((None), dtype=tf.dtypes.int32),
            tf.TensorSpec((None), dtype=tf.dtypes.int32),
            tf.TensorSpec((None), dtype=tf.dtypes.int32),
            tf.TensorSpec((None), dtype=tf.dtypes.int32)
        ]

        # Compile the update function using the input signature for speedy code.
        self.step_op = tf.function(train_step, input_signature=predictor.input_signature + self.input_signature)

    #supervised training using reference alignment
    #randomly select a center columns with a window of adjacent columns within
    #a certain radius (if existing)
    def train(self, msa):
        radius = self.config["adjacent_column_radius"]#np.random.randint(2, self.config["adjacent_column_radius"])
        c = np.random.randint(msa.alignment_len)
        lb = max(0, c - radius)
        ub = min(msa.alignment_len-1, c + radius)
        seq_g, col_g, priors, target_col_ids, gaps_in, gaps_start, gaps_end = self.predictor.get_window_sample(msa, lb, ub, ub -lb +1)#self.config["num_col"])
        return self.step_op(seq_g, col_g, priors, target_col_ids, gaps_in, gaps_start, gaps_end)
