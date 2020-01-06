

param = {
    "train_mp_iterations" : 40,
    "test_mp_iterations" : 100,
    "learning_rate" : 1e-5,
    "optimizer_momentum": 0.9,

    "enc_node_layer_s" : [128]*2,
    "enc_edge_layer_s" : [64]*2,
    "enc_globals_layer_s" : [128]*2,

    "seq_core_node_layer_s" : [128]*4,
    "seq_core_edge_layer_s" : [64]*4,
    "seq_core_globals_layer_s" : [128]*4,

    "pattern_core_node_layer_s" : [128]*4,
    "pattern_core_edge_layer_s" : [64]*4,
    "pattern_core_globals_layer_s" : [128]*4,

    "anchor_graph_edge_layer_s" : [64]*2,

    "rp_decoder_layer_s" : [128]*2
}
