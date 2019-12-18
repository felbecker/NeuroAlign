import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
import sys
sys.path.append('./ProcessSeq')

import AnchorSet
import PatternSet

parser = argparse.ArgumentParser(description='Convert a fasta file of sequences to a graph file.')
parser.add_argument("-i", type=int, default=0, help="number of alignment file")
parser.add_argument("-r", type=int, default=10, help="the kmer radius")
parser.add_argument("-t", type=int, default=21, help="the threshold for k-mer matching")
parser.add_argument("-s", type=str, default="blosum62.txt", help="the underlying scoring matrix")
parser.add_argument("-a", type=int, default=200, help="maximum number of anchors allowed")
parser.add_argument("-minrow", type=int, default=-1, help="minimum number of edges to build a row")
parser.add_argument("-threads", type=int, default=4, help="number of threads to use")
parser.add_argument("-draw", type=int, default=1, help="0 = render graphs, 1 = dont")
parser.add_argument("-pattern_graph", type=int, default=0, help="if set to 1, the pattern graph is drawn instead of the AL graphs")
args = parser.parse_args()

####
# Outline:
# 1) read sequences
# 2) compute k-mer-based anchors
# 3) filter rows of a minimum length, remove anchors not contained in a minimum length row
# 4) contract rows to anchieve a new, reduced set of anchors
# 5A) find alignment columns; connect all anchors in a column with "positive" connections
# 5B) alt define all incident anchors that are not locally inconsitent as "positive" connections
# 6) find all local inconsistencies; connect all pairs of locally inconsistent anchors with a "negative" connection
# 7) train and evaluate a GNN passing messages along positive/negative connections; use the global attribute to learn a representation
#    that is able to solve global inconsistencies

scoring = AnchorSet.ScoringMatrix(args.s)

filename = "A"+"{0:0=4d}".format(args.i)

instance = AnchorSet.MSAInstance("./data/"+filename+".fasta", True)

if args.t == -1:
    threshold = AnchorSet.sample_threshold(instance, args.r)
    print("sampled threshold: ", threshold)
else:
    threshold = args.t

start = time.time()
anchors = AnchorSet.anchor_set_kmer_threshold(instance, scoring, args.r, threshold, args.threads)
end = time.time()
print("time for edge search: ", end - start)
AnchorSet.read_solution("./data/"+filename+".fasta", anchors)

#save to file and load again:
#AnchorSet.prior_prob(anchors)
#locinc = AnchorSet.compute_local_inconsistencies(anchors)
# print(anchors.solution)
# anchors.to_file("./TEST")
# anchors2 = ProcessSeq.anchor_set_from_file("./TEST")
# print(anchors2.solution)

rows = AnchorSet.build_alignment_rows(anchors)

if args.minrow == -1:
    minrow = AnchorSet.sample_min_row(rows)
    print("sampled minrow: ", minrow)
else:
    minrow = args.minrow

rows = [r for r in rows if len(r) >= minrow]

#print(rows)
#anchors_reduced = AnchorSet.reduce_anchors_to_rows(anchors, rows, minrow)

start = time.time()
anchors_row_contraction = AnchorSet.row_contraction(instance, anchors, rows, minrow)
end = time.time()

print("time for row contraction: ", end - start)

print("anchors total: ", anchors.anchor_data.shape[0])
print("anchors after contraction: ", anchors_row_contraction.anchor_data.shape[0])

anchors_row_contraction = AnchorSet.kBestAnchors(instance, anchors_row_contraction, args.a)

AnchorSet.read_solution("./data/"+filename+".fasta", anchors_row_contraction)
patterns = PatternSet.find_patterns(anchors_row_contraction)
PatternSet.compute_targets(patterns)


if args.draw==1:

    fig = plt.figure(1, figsize=(20, 6))
    fig.clf()

    if args.pattern_graph == 0:

        ax = fig.add_subplot(1, 3, 1)
        anchors.draw_alignment_graph(ax)

        #ax = fig.add_subplot(1, 3, 2)
        #anchors_reduced.draw_alignment_graph(ax)

        ax = fig.add_subplot(1, 3, 3)
        anchors_row_contraction.draw_alignment_graph(ax)

    else:

        ax = fig.add_subplot(1, 1, 1)
        patterns.draw_pattern_graph(ax)


    plt.draw()
    plt.show()
