HIDDEN_DIM = 50
LAYER_DIM = 100

#NeuroAlign parameters
config = {
    #number of sequentially applied core networks (each with unique parameters)
    "num_nr_core" : 1,
    #number of iterations inside each core network (shared parameters) during training
    "train_mp_iterations" : 10,
    #number of iterations inside each core network (shared parameters) during testing
    "test_mp_iterations" : 10,

    #training performance and logging
    "learning_rate" : 1e-5,
    "num_training_iteration": 10000,
    "batch_size": 50,
    "savestate_milestones": 10,
    "l2_regularization" : 0,#1e-7,

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
    "column_net_node_layers" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],
    "column_net_global_layers" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],

    #layers in the decoding networks
    "seq_dec_node_layer_s" : [HIDDEN_DIM],
    "mem_dec_node_layer_s" : [HIDDEN_DIM, LAYER_DIM, LAYER_DIM, HIDDEN_DIM],
    "mem_dec_global_layer_s" : [HIDDEN_DIM],

    "len_alphabet" : 4
}
