import os
import numpy as np
import copy


#reads sequences as fasta file and converts them to inputs and targets interpretable by the NeuroAlign model
class Instance:
    def __init__(self, filename, alphabet, gaps = True, contains_lower_case = False):
        self.filename = filename
        self.alphabet = alphabet
        self.seq_ids = []
        self.valid = self.read_seqs(filename, gaps, contains_lower_case)
        if self.valid and gaps:
            self.compute_targets()

    def read_seqs(self, filename, gaps, contains_lower_case):
        print("reading file ", self.filename)
        #read seqs as strings
        _, file_extension = os.path.splitext(filename)
        with open(filename) as f:
            content = f.readlines()
        self.raw_seq = []
        seq_open = False
        for line in content:
            line = line.strip()
            if len(line)>0:
                if line[0]=='>':
                    seq_open = True
                    self.seq_ids.append(line[1:])
                elif seq_open:
                    self.raw_seq.append(line)
                    seq_open = False
                else:
                    self.raw_seq[-1] += line

        #convert to numpy arrays
        for seq in self.raw_seq:
            if not seq.find("/") == -1:
                return False

        if gaps:
            self.alignment_len = len(self.raw_seq[0])

        self.raw_seq = [s.replace('.','-') for s in self.raw_seq] #treat dots as gaps
        for i,c in enumerate(self.alphabet):
            self.raw_seq = [s.replace(c,str(i)+' ') for s in self.raw_seq]
            if contains_lower_case:
                self.raw_seq = [s.replace(c.lower(),str(i)+' ') for s in self.raw_seq]

        #can store sequences with gaps as matrix
        if gaps:
            self.ref_seq = copy.deepcopy(self.raw_seq)
            self.ref_seq = [s.replace('-',str(len(self.alphabet))+' ') for s in self.ref_seq]
            self.ref_seq = np.reshape(np.fromstring("".join(self.ref_seq), dtype=int, sep=' '), (len(self.ref_seq), self.alignment_len))

        self.raw_seq = [s.replace('-','') for s in self.raw_seq]
        self.raw_seq = [np.fromstring(s, dtype=int, sep=' ') for s in self.raw_seq]
        self.seq_lens = [s.shape[0] for s in self.raw_seq]
        self.num_columns = 2*max(self.seq_lens)
        self.total_len = sum(self.seq_lens)
        return True



    def compute_targets(self):
        #a mapping from raw position to position in the reference solution (sequences with gaps)
        cumsum = np.cumsum(self.ref_seq != len(self.alphabet), axis=1) #A-B--C -> 112223
        diff = np.diff(np.insert(cumsum, 0, 0.0, axis=1), axis=1) #112223 -> 0112223 -> [[(i+1) - i]] -> 101001
        diff_where = [np.argwhere(diff[i,:]).flatten() for i in range(diff.shape[0])]
        self.gap_lengths = np.concatenate([np.diff(np.concatenate([-np.ones(1), d, self.alignment_len*np.ones(1)]))-1 for d in diff_where]).flatten()
        self.membership_targets = np.concatenate(diff_where).flatten()

        #also compute a mapping for each sequence from alignment column to the last occuring index
        #self.col_to_seq[i-1] == self.col_to_seq[i]  <->  gap at i
        self.col_to_seq = cumsum - 1

        #for each symbol in the alphabet, count the relative number of occurences per column
        self.rel_occ_per_column = np.zeros((self.alignment_len, len(self.alphabet)+1))
        for s_id in range(len(self.alphabet)+1):
            self.rel_occ_per_column[:,s_id] = np.sum(self.ref_seq == s_id, axis = 0)
        self.rel_occ_per_column /= self.ref_seq.shape[0]


    #computes recall and precision of the edge predictions
    #over all possible pairs of positions in the sequences
    # "true positive" = aligned in NR and reference
    # "true negative" = not aligned in NR and reference
    # "false positive" = aligned in NR but not in reference
    # "false negative" = not aligned in NR but in reference
    def recall_prec(self, choices):

        tp, tn, fp, fn = 0,0,0,0

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


#takes a vector of column indices and a MSA instance and outputs a fasta file
#the column indices per sequence have to be strictly increasing
def column_pred_to_fasta(msa, cols, dir):
    ncol = np.max(cols)+1
    lsum = 0
    file = dir+os.path.basename(msa.filename)
    with open(file,"w") as f:
        for id,l,raw in zip(msa.seq_ids, msa.seq_lens, msa.raw_seq):
            seq_with_gaps = ""
            for i,c in enumerate(cols[lsum:(lsum+l)]):
                seq_with_gaps += "-"*int(c - len(seq_with_gaps))
                seq_with_gaps += msa.alphabet[raw[i]]
            seq_with_gaps += "-"*int(ncol - len(seq_with_gaps))
            f.write(">"+id+"\n")
            f.write(seq_with_gaps+"\n")
            lsum += l
