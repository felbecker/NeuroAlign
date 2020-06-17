#NeuroAlign parameter configuration
config = {

    "type" : "protein",  #currently supports nucleotide or protein

    #"num_col" : 250,

    #number of sequentially applied core networks (each with unique parameters)
    "num_kernel" : 1,

    #iteration counts for the different components
    "train_iterations" : 10,
    "test_iterations" : 10,

    #training performance and logging
    "learning_rate" : 1e-3,
    "num_training_iteration" : 2000,
    "batch_size": 10,
    "savestate_milestones": 10,
    "l2_regularization" : 1e-10,
    "adjacent_column_radius" : 3000,
    "window_uniform_radius" : 10,

    #hidden dimension for the latent representations for each alphabet symbol
    #for simplicity also used for the representations of their interactions (edges) and their global representation
    "alphabet_network_layers" : [50, 50],

    #hidden dimension for the latent representations for each sequence position
    #for simplicity also used for the representations of the forward edges the along sequences and
    #the global representation for each sequence
    "seq_latent_dim" : 40,

    "seq_global_dim" : 100,

    "seq_net_node_layers" : [40,40],
    "seq_net_edge_layers" : [40,40],
    "seq_net_global_per_seq_layers" : [40,40],
    "seq_global_layers" : [40,40],

    "alphabet_to_sequence_layers" : [40],
    "columns_to_sequence_layers" : [40],

    #hidden dimension for the latent representation of the global property of each column
    "col_latent_dim" : 40,

    "column_net_node_layers" : [40,40],
    "column_net_global_layers" : [40,40],
    "column_decode_node_layers" : [40,40],

    "sequence_to_columns_layers" : [40]
}
