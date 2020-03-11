HIDDEN_DIM = 50
LAYER_DIM = 100

#NeuroAlign parameters
config = {
    "train_mp_iterations" : 10,
    "test_mp_iterations" : 20,
    "learning_rate" : 2e-5,
    "num_training_iteration": 1000,
    "batch_size": 1,
    "savestate_milestones": 5,
    "l2_regularization" : 0.0001,

    #layers in the sequence encoding network
    "seq_enc_edge_layer_s" : [HIDDEN_DIM],
    "seq_enc_node_layer_s" : [HIDDEN_DIM],
    "seq_enc_globals_layer_s" : [HIDDEN_DIM],

    #layers in the pattern encoding network
    "mem_enc_node_layer_s" : [HIDDEN_DIM],
    "mem_enc_globals_layer_s" : [HIDDEN_DIM],

    #layers in the core network for the sequence graph
    "seq_net_edge_layers" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],
    "seq_net_node_layers" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],
    "seq_net_global_layers" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],

    #layers in the core network for the pattern graph
    "column_net_node_layers" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],
    "column_net_global_layers" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],

    #layers in the decoding networks
    "seq_dec_node_layer_s" : [HIDDEN_DIM],
    "mem_dec_node_layer_s" : [HIDDEN_DIM],
    "mem_dec_global_layer_s" : [HIDDEN_DIM]
}
