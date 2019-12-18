#third party imports
import argparse
import multiprocessing
import sys
import numpy as np

#project imports
sys.path.append('./ProcessSeq')
import AnchorSet

parser = argparse.ArgumentParser(description='Computes jaccard indices with reference alignments for each of the listed 3rd party aligners')
parser.add_argument("-test_set", type=str, default="10", help="filename of a textfile that lists all alignments to tests (linewise filepaths to fasta files)")
parser.add_argument("-num_thread", type=int, default=4, help="number of processes to compute the scores in parallel")
parser.add_argument("-aligner", type=str, default="", help="scripte assumes that there exists a directory called 'aligner' in the working directory containing fasta files of computed alignments")
args = parser.parse_args()

def eval_one(f):
    print("Eval ", f)
    anchor_set = AnchorSet.anchor_set_from_file(f)
    unwrapped_anchors = AnchorSet.unwrap_anchor_set(anchor_set)
    name = f.split('/')[1]
    AnchorSet.read_solution("./"+args.aligner+"/"+name+".fasta", unwrapped_anchors)
    aligner_solution = unwrapped_anchors.solution
    AnchorSet.read_solution("./data/"+name+".fasta", unwrapped_anchors)
    reference_solution = unwrapped_anchors.solution
    score = AnchorSet.jaccard_index(aligner_solution, reference_solution)
    print("score of ", f, ":", score)
    return aligner_solution, reference_solution

filenames = []
with open(args.test_set, "r") as f:
    for line in f:
        filenames.append(line.strip())

#data
pool = multiprocessing.Pool(args.num_thread)
result = pool.map(eval_one, filenames)
global_aligner = np.zeros((0))
global_ref = np.zeros((0))
for (asol, rsol) in result:
    global_aligner = np.concatenate((global_aligner, asol), axis=0)
    global_ref = np.concatenate((global_ref, rsol), axis=0)
print("global score for", args.aligner, AnchorSet.jaccard_index(global_aligner, global_ref))
