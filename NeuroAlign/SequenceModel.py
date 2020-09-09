import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import numpy as np


LSTM_DIM = 256

def make_lstm_model(layersizes):
    return lambda: snt.DeepRNN([snt.LSTM(s) for s in layersizes])



class LSTMSequenceModel(gn._base.AbstractModule):
    def __init__(self, name = "LSTMSequenceModel"):
        super(LSTMSequenceModel, self).__init__(name=name)
        self.forwardLSTM = make_lstm_model([256,256])
        self.backwardLSTM = make_lstm_model([256,256])


    def _build(self, graph, hidden_graph):
        sequence_output, final_state = tf.keras.layers.RNN(core, sequence_input, dtype=tf.float32)
