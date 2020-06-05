#NeuroAlign parameter configuration
config = {

    "type" : "protein",  #currently supports nucleotide or protein

    #"num_col" : 250,

    #number of sequentially applied core networks (each with unique parameters)
    "num_col_kernel" : 1,

    #iteration counts for the different components
    "train_col_iterations" : 3,
    "train_alpha_iterations_per_col" : 1,
    "train_seq_iterations_per_col" : 1,
    "test_col_iterations" : 3,
    "test_alpha_iterations_per_col" : 1,
    "test_seq_iterations_per_col" : 1,

    #training performance and logging
    "learning_rate" : 1e-4,
    "num_training_iteration" : 2000,
    "batch_size": 50,
    "savestate_milestones": 50,
    "l2_regularization" : 0,#1e-8,
    "adjacent_column_radius" : 3000,
    "window_uniform_radius" : 20,

    #hidden dimension for the latent representations for each alphabet symbol
    #for simplicity also used for the representations of their interactions (edges) and their global representation
    "alphabet_latent_dim" : 100,

    "alphabet_net_node_layers" : [100,100],
    "alphabet_net_edge_layers" : [100,100],
    "alphabet_net_global_layers" : [100,100],

    "alphabet_to_sequence_layers" : [100],
    "alphabet_to_column_layers" : [100],

    #hidden dimension for the latent representations for each sequence position
    #for simplicity also used for the representations of the forward edges the along sequences and
    #the global representation for each sequence
    "seq_latent_dim" : 100,

    "seq_net_node_layers" : [100,100],
    "seq_net_edge_layers" : [100,100],
    "seq_net_global_layers" : [100,100],

    "sequence_to_column_layers" : [100],
    "sequence_to_alphabet_layers" : [100],

    #hidden dimension for the latent representation of the global property of each column
    "col_latent_dim" : 150,

    "column_net_node_layers" : [150,150],
    "column_net_global_layers" : [150,150],
    "column_encode_node_layers" : [150],
    "column_decode_node_layers" : [150],

    "column_to_sequence_layers" : [150],
    "column_to_alphabet_layers" : [150]
}
