#NeuroAlign parameter configuration
config = {

    "type" : "protein",  #currently supports nucleotide or protein

    "num_col" : 20,

    #number of sequentially applied core networks (each with unique parameters)
    "num_col_kernel" : 1,

    #number of iterations inside each core network (shared parameters) during training
    "train_mp_iterations" : 20,
    #number of iterations inside each core network (shared parameters) during testing
    "test_mp_iterations" : 20,

    #training performance and logging
    "learning_rate" : 1e-3,
    "num_training_iteration" : 2000,
    "batch_size": 50,
    "savestate_milestones": 10,
    "l2_regularization" : 0,#1e-5,
    "adjacent_column_radius" : 3000,
    "window_uniform_radius" : 50,

    #hidden dimension for the latent representations for each alphabet symbol
    #for simplicity also used for the representations of their interactions (edges) and their global representation
    "alphabet_latent_dim" : 32,

    "alphabet_net_node_layers" : [32, 32],
    "alphabet_net_edge_layers" : [32, 32],
    "alphabet_net_global_layers" : [32],

    "alphabet_to_sequence_layers" : [32],
    "alphabet_to_column_layers" : [32],

    #hidden dimension for the latent representations for each sequence position
    #for simplicity also used for the representations of the forward edges the along sequences and
    #the global representation for each sequence
    "seq_latent_dim" : 32,

    "seq_net_node_layers" : [32, 32],
    "seq_net_edge_layers" : [32, 32],
    "seq_net_global_layers" : [32, 32],

    "sequence_to_column_layers" : [32],
    "sequence_to_alphabet_layers" : [32],

    #hidden dimension for the latent representation of the global property of each column
    "col_latent_dim" : 32,

    "column_net_node_layers" : [32, 32],
    "column_net_global_layers" : [32, 32],
    "column_encode_node_layers" : [32],
    "column_decode_node_layers" : [32, 32, 32, 32],

    "column_to_sequence_layers" : [32],
    "column_to_alphabet_layers" : [32]
}
