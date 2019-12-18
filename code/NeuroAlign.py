import tensorflow as tf
import numpy as np

#project imports
import ProteinGraphNN
import ProcessSeq

model_num = 50

parser = argparse.ArgumentParser(description='Predicts high quality anchors for a given fasta file.')
parser.add_argument("-f", type=str, default="", help="a fasta file")
args = parser.parse_args()

if len(args.f) == "":
    print("Please provide a fasta file.")
    sys.exit()

tf.reset_default_graph()
graphNN = ProteinGraphNN.ProteinGNN(0.1, 10, 10, input_graphs[0], target_graphs[0])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "./model_proteinGNN/model_"+str(model_num)+".ckpt")

with open("model_proteinGNN/test_instances.txt", "r") as f:
    test_files = f.readlines()

print("Data reading and preprocessing. This may take some time...")

def load_one(filename):
    #print("Reading ", filename)
    anchor_set = ProcessSeq.anchor_set_from_file(filename)
    local_inconsistencies = ProcessSeq.compute_local_inconsistencies(anchor_set)
    input_dict, target_dict = ProteinGraphNN.anchor_set_to_input_target_dicts(anchor_set, local_inconsistencies)
    return input_dict, target_dict

#data
pool = multiprocessing.Pool(data_threads)
result = pool.map(load_one, test_files)
result = [r for r in result]
input_graphs, target_graphs = zip(*result)
input_graphs, target_graphs = list(input_graphs), list(target_graphs)

input_graphs = [utils_np.data_dicts_to_graphs_tuple([g]) for g in input_graphs]
target_graphs = [utils_np.data_dicts_to_graphs_tuple([g]) for g in target_graphs]

##compute predictions for each test file
##compare predictions to precomputed alignments (e.g. form clustalw) using jaccard_index
