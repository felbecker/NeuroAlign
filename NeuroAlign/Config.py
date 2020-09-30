STATE_DIM = 32
HIDDEN_LAYER_DIM = 64

#NeuroAlign parameter configuration
config = {

    "type" : "nucleotide",  #currently supports nucleotide or protein

    #"num_col" : 250,

    #number of sequentially applied core networks (each with unique parameters)
    "num_kernel" : 2,

    #iteration counts for the different components
    "train_iterations" : 20,
    "test_iterations" : 20,

    #training performance and logging
    "learning_rate" : 2e-5,
    "num_training_iteration" : 20000,
    "batch_size": 10,
    "savestate_milestones": 500,
    "l2_regularization" : 1e-10,
    "adjacent_column_radius" : 30,
    "window_uniform_radius" : 10,
    "final_iteration_loss_weight" : 1, #final iteration is weighted xxx times as much as any intermediate iteration
    "lambda_rp" : 1.0,
    "lambda_gap" : 1.0,
    "lambda_res_dist" : 1.0,
    "decode_batches_test" : 1,
    "decode_batches_train" : 1,

    #hidden dimension for the latent representations for each sequence position
    #for simplicity also used for the representations of the forward edges the along sequences and
    #the global representation for each sequence
    "seq_latent_dim" : STATE_DIM,

    "seq_global_dim" : 5*STATE_DIM,

    "encoder" : [HIDDEN_LAYER_DIM],

    "seq_net_node_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],
    "seq_net_edge_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],
    "seq_net_global_per_seq_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],
    "seq_global_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],

    "alphabet_to_sequence_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],
    "columns_to_sequence_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],

    #hidden dimension for the latent representation of the global property of each column
    "col_latent_dim" : STATE_DIM,

    "column_net_node_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],
    "column_net_global_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],
    "column_net_edge_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],
    "column_decode_node_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM],

    "sequence_to_columns_layers" : [HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM]
}
