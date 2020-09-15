import argparse
import MSA
import Model
import numpy as np
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

from Config import config
import Postprocessing

parser = argparse.ArgumentParser(description='Tests the latest NeuroAlign model.')
parser.add_argument("-n", type=int, default=200, help="number of testing examples")
parser.add_argument("-dir", type=str, default="", help="directory with data files")
parser.add_argument("-w", action='store_true', help="write output fasta files and files for the intermediate outputs (memberships, relative positions, gap lengths)")
parser.add_argument("-nogaps", action='store_true', help="input files have no gaps")
parser.add_argument("-col_max", action='store_true', help="enables greedy col max postprocessing (fastest)") #this output is not suitable for constructing complete alignments, only for SP score
parser.add_argument("-seq_consistent", action='store_true', help="enables greedy seq-level consistent postprocessing (fast)")
parser.add_argument("-fully_consistent", action='store_true', help="enables greedy fully consistent postprocessing (slow)")
args = parser.parse_args()


if args.w:
    try:
        os.rmdir("./NeuroAlign_out")
    except OSError:
        print ("Can not remove test directory")
    try:
        os.mkdir("./NeuroAlign_out")
    except OSError:
        print ("Can not remove test directory")

class PostProc:
    def __init__(self, callback):
        self.prec_sum = 0
        self.rec_sum = 0
        self.callback = callback
        self.runs = 0

    def run(self, msa, mem):
        cols = self.callback(msa, mem)
        if not args.nogaps:
            p,r = m.recall_prec(cols.flatten())
            print("prec=",p, " recall=",r)
            self.prec_sum += p
            self.rec_sum += r
            self.runs += 1
        return cols

    def print(self, name):
        print(name + " precision=", self.prec_sum/self.runs, "recall=", self.rec_sum/self.runs)


predictor = None #lazy instantiation with the first input example

col_max_postproc = PostProc(Postprocessing.max_likely)
seq_consistent_postproc = PostProc(Postprocessing.seq_consistent)
fully_consistent_postproc = PostProc(Postprocessing.fully_consistent)

alphabet = ['A', 'C', 'G', 'T'] if config["type"] == "nucleotide" else ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X']
for i in range(1,args.n+1):

    filename = "/A"+"{0:0=4d}".format(i)
    filepath = args.dir + filename + ".fa"
    m = MSA.Instance(filepath, alphabet, not args.nogaps)
    if not m.valid:
        continue

    print("file:", m.filename)

    if predictor == None:
        #instantiate the predictor with the first input example m
        predictor = Model.NeuroAlignPredictor(config, m)
        predictor.load_latest()

    mem, rp, gaps = predictor.predict(m)

    if args.w:
        np.savetxt("./NeuroAlign_out"+filename + ".mem", mem)
        np.savetxt("./NeuroAlign_out"+filename + ".rp", rp)
        np.savetxt("./NeuroAlign_out"+filename + ".gap", gaps)

    if args.col_max:
        cols = col_max_postproc.run(m, mem)
    if args.fully_consistent:
        cols = fully_consistent_postproc.run(m, mem)
    if args.seq_consistent or (not args.col_max) and (not args.fully_consistent):
        cols = seq_consistent_postproc.run(m, mem)

    if args.w:
        MSA.column_pred_to_fasta(m, cols, "./NeuroAlign_out/")

if args.col_max:
    col_max_postproc.print("col max")
if args.seq_consistent or (not args.col_max) and (not args.fully_consistent):
    seq_consistent_postproc.print("seq consistent")
if args.fully_consistent:
    fully_consistent_postproc.print("fully consistent")
