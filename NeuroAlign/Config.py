#NeuroAlign parameter configuration
config = {

    "type" : "protein",  #currently supports nucleotide or protein

    #"num_col" : 250,

    #number of sequentially applied core networks (each with unique parameters)
    "num_kernel" : 1,

    #iteration counts for the different components
    "train_iterations" : 20,
    "test_iterations" : 20,

    #training performance and logging
    "learning_rate" : 1e-4,
    "num_training_iteration" : 2000,
    "batch_size": 10,
    "savestate_milestones": 100,
    "l2_regularization" : 1e-10,
    "adjacent_column_radius" : 1000,
    "window_uniform_radius" : 10,

    #hidden dimension for the latent representations for each sequence position
    #for simplicity also used for the representations of the forward edges the along sequences and
    #the global representation for each sequence
    "seq_latent_dim" : 32,

    "seq_global_dim" : 32,

    "seq_net_node_layers" : [64, 64],
    "seq_net_edge_layers" : [64, 64],
    "seq_net_global_per_seq_layers" : [64, 64],
    "seq_global_layers" : [64, 64],

    "alphabet_to_sequence_layers" : [64],
    "columns_to_sequence_layers" : [64],

    #hidden dimension for the latent representation of the global property of each column
    "col_latent_dim" : 32,

    "column_net_node_layers" : [64, 64],
    "column_net_global_layers" : [64, 64],
    "column_decode_node_layers" : [64, 64],

    "sequence_to_columns_layers" : [64]
}
