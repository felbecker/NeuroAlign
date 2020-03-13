import os
import numpy as np
import copy


alphabet = ['A', 'C', 'G', 'T']
gap_id = len(alphabet)

#reads sequences as fasta file and converts them to inputs and targets interpretable by the NeuroAlign model
class Instance:
    def __init__(self, filename):
        self.read_seqs(filename)
        self.compute_inputs()
        self.compute_targets()

    def read_seqs(self, filename):
        #read seqs as strings
        _, file_extension = os.path.splitext(filename)
        with open(filename) as f:
            content = f.readlines()
        self.ref_seq = []
        seq_open = False
        for line in content:
            line = line.strip()
            if len(line)>0:
                if line[0]=='>':
                    seq_open = True
                elif seq_open:
                    self.ref_seq.append(line)
                    seq_open = False
                else:
                    self.ref_seq[-1] += line

        #convert to numpy arrays
        for i,c in enumerate(alphabet):
            self.ref_seq = [s.replace(c,str(i)+' ') for s in self.ref_seq]
        self.raw_seq = copy.deepcopy(self.ref_seq)
        self.ref_seq = [s.replace('-',str(gap_id)+' ') for s in self.ref_seq]
        self.raw_seq = [s.replace('-','') for s in self.raw_seq]
        #can store this as matrix since seq + gaps have uniform length
        self.ref_seq = np.reshape(np.fromstring("".join(self.ref_seq), dtype=int, sep=' '), (len(self.ref_seq), int(len(self.ref_seq[0])/2)))
        self.raw_seq = [np.fromstring(s, dtype=int, sep=' ') for s in self.raw_seq]

        self.seq_lens = [s.shape[0] for s in self.raw_seq]
        self.alignment_len = len(self.ref_seq[0])
        self.num_columns = 2*max(self.seq_lens)
        self.total_len = sum(self.seq_lens)



    def compute_inputs(self):
        #convert to binary alphabet
        #for each node, we store a one hot encoding of the respective symbol and the relative raw seq position
        self.nodes = np.zeros((self.total_len, len(alphabet)+1), dtype=np.float32)
        node_iter = 0
        for seq in self.raw_seq:
            for i,s in enumerate(seq):
                self.nodes[node_iter, s] = 1 #onehot
                self.nodes[node_iter, len(alphabet)] = i / seq.shape[0] #relative pos
                node_iter += 1

        #compute forward edges
        self.forward_senders = []
        self.forward_receivers = []
        sum_l = 0
        for l in self.seq_lens:
            self.forward_senders.extend(range(sum_l, sum_l+l-1))
            self.forward_receivers.extend(range(sum_l+1, sum_l+l))
            sum_l += l



    def compute_targets(self):
        #a mapping from raw position to position in the reference solution (sequences with gaps)
        cumsum = np.cumsum(self.ref_seq != gap_id, axis=1) #A-B--C -> 112223
        diff = np.diff(np.insert(cumsum, 0, 0.0, axis=1), axis=1) #112223 -> 0112223 -> [[(i+1) - i]] -> 101001
        self.membership_targets = np.concatenate([np.argwhere(diff[i,:]) for i in range(diff.shape[0])]).flatten()
        self.node_rp_targets = np.reshape(self.membership_targets / self.alignment_len, (-1,1))
        #for each symbol in the alphabet, count the relative number of occurences per column
        self.rel_occ_per_column = np.zeros((self.alignment_len, len(alphabet)+1))
        for s_id in range(len(alphabet)+1):
            self.rel_occ_per_column[:,s_id] = np.sum(self.ref_seq == s_id, axis = 0)
        self.rel_occ_per_column /= self.ref_seq.shape[0]

    #computes recall and precision of the edge predictions
    #over all possible pairs of positions in the sequences
    # "true positive" = aligned in NR and reference
    # "true negative" = not aligned in NR and reference
    # "false positive" = aligned in NR but not in reference
    # "false negative" = not aligned in NR but in reference
    def recall_prec(self, predictions):

        tp, tn, fp, fn = 0,0,0,0
        choices = np.argmax(predictions, axis=1).flatten()

        for i, (target_i, choice_i) in enumerate(zip(self.membership_targets, choices)):
            for j, target_j, choice_j in zip(range(i, len(self.membership_targets)), self.membership_targets[i:], choices[i:]):
                if target_i == target_j:
                    if choice_i == choice_j:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if choice_i == choice_j:
                        fp += 1
                    else:
                        tn += 1

        prec = tp/(tp+fp) if tp+fp > 0 else 1
        rec = tp/(tp+fn) if tp+fn > 0 else 1

        return prec, rec
