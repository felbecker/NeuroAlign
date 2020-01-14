import ProcessSeq
import numpy as np
import argparse
import sys
sys.path.append('./ProcessSeq')
import AnchorSet

parser = argparse.ArgumentParser(description='Computes edge sets of alignment graphs for all ref alignments in ./data')
parser.add_argument("-r", type=int, default=7, help="the kmer radius")
parser.add_argument("-t", type=int, default=-1, help="treshold")
parser.add_argument("-s", type=str, default="blosum62.txt", help="the underlying scoring matrix")
parser.add_argument("-minrow", type=int, default=-1, help="minimum number of edges to build a row")
parser.add_argument("-a", type=int, default=200, help="maximum number of anchors allowed")
args = parser.parse_args()

num_alignments = 1509

NUM_THREAD = 20

scoring = AnchorSet.ScoringMatrix(args.s)

#compute alignment graphs for all ref alignments
for i in range(num_alignments):

    print(i)

    av_sol_sum = 0.0
    av_num_edge_sum = 0

    name = "A"+"{0:0=4d}".format(i)
    instance = AnchorSet.MSAInstance("../data/data_unaligned/"+name+".fasta", True)

    skip = False
    for s in instance.seq:
        if '/' in s:
            print("/ found, skipping ", name)
            skip = True
            break
    if skip:
        continue

    skip = False
    for s in instance.seq:
        if len(s) < 3*args.r+1:
            print("Sequence too short, skipping ", name)
            skip = True
            break
    if skip:
        continue

    if args.t == -1:
        threshold = AnchorSet.sample_threshold(instance, args.r)
    else:
        threshold = args.t
    anchors = AnchorSet.anchor_set_kmer_threshold(instance, scoring, args.r, threshold, NUM_THREAD)
    AnchorSet.read_solution("../data/data/"+name+".fasta", anchors)
    rows = AnchorSet.build_alignment_rows(anchors)
    if len(rows) > 0:
        if args.minrow == -1:
            minrow = AnchorSet.sample_min_row(rows)
        else:
            minrow = args.minrow
        rows = [r for r in rows if len(r) >= minrow]
        anchors_row_contraction = AnchorSet.row_contraction(instance, anchors, rows, minrow)
        anchors_row_contraction = AnchorSet.kBestAnchors(instance, anchors_row_contraction, args.a)
        anchors_row_contraction.to_file("../data/anchors_"+str(args.r)+"_"+str(threshold)+"_"+str(args.a)+"/"+name)
    else:
        print("No fitting rows found: ", name)

    if anchors_row_contraction.loaded:
        av_sol_sum += np.sum(anchors_row_contraction.solution)/len(anchors_row_contraction.solution)
        av_num_edge_sum += anchors_row_contraction.anchor_data.shape[0]
