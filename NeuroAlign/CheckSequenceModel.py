import SequenceModel
import os
import numpy as np
from pdb_class import pdb

##################################################################################################
##################################################################################################

sequence_model = SequenceModel.make_model()
if os.path.isfile(SequenceModel.CHECKPOINT_PATH+"/saved_model.pb"):
    sequence_model.load_weights(SequenceModel.CHECKPOINT_PATH+"/variables/variables")
    print("Sequence model loaded.", flush=True)
else:
    print("No sequence model found.", flush=True)
    quit()

##################################################################################################
##################################################################################################

# #load the data
# msa = MSA.Instance("Pfam-80-500.fasta", SequenceModel.ALPHABET, gaps=False)
#
# if not msa.valid:
#     print("Invalid data.")
#     quit()
#
# np.random.seed(0)
#
# indices = np.array(range(len(msa.raw_seq)))
# np.random.shuffle(indices)
# train, test = np.split(indices, [int(len(msa.raw_seq)*(1-SequenceModel.VALIDATION_SPLIT))])

##################################################################################################
##################################################################################################

pdbs = ["5GQE", "5GQE", "5GQE"]

NUM_CLASSES = 3

acc_sum = 0
acc_class_sum = [0 for _ in range(NUM_CLASSES)]
for id in pdbs:
    protein = pdb(id, work_dir= "./pdb/") # in "work_dir" wird der Ordner 5GQE mit dem .pdb file erstellt

    raw_seq = protein.return_seq()
    int_seq = raw_seq
    for i,aa in enumerate(SequenceModel.ALPHABET):
        int_seq = int_seq.replace(aa, str(i)+' ')
        int_seq = int_seq.replace(aa.lower(), str(i)+' ')
    seq_arr = np.fromstring(int_seq, dtype=int, sep=' ')
    struct_arr = np.array(protein.return_structure(), dtype=np.int32)

    seq = np.zeros((1, len(raw_seq), len(SequenceModel.ALPHABET)), dtype=np.float32)
    seq[0,np.arange(len(raw_seq)), seq_arr] = 1

    out = sequence_model(seq)

    correct = (np.argmax(out, axis=-1) == np.argmax(seq, axis=-1))

    acc_sum += np.sum(correct)/len(raw_seq)

    for i in range(NUM_CLASSES):
        class_subset = struct_arr == i
        num = np.sum(class_subset)
        num_correct = np.sum(class_subset*correct)
        acc_class_sum[i] += num_correct/num


print("overall accuracy: ", acc_sum/len(pdbs))
for i,csum in enumerate(acc_class_sum):
    print("class ", i, " accuracy: ", csum/len(pdbs))
