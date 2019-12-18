import argparse
import os

import ProcessSeq

parser = argparse.ArgumentParser(description='Generates a meta file for a directory of anchor sets. The meta file is used for data normalization.')
parser.add_argument("-r", type=int, default=10, help="the kmer radius")
args = parser.parse_args()

num_alignments = 1509

filenames = []
for i in range(num_alignments):
    #files can be missing when no anchors were found for a parameter choice r, t, minrow
    #skip these files
    if os.path.isfile("anchors_"+str(args.r)+"/A"+"{0:0=4d}".format(i)+".anchors") and os.path.isfile("anchors_"+str(args.r)+"/A"+"{0:0=4d}".format(i)+".meta"):
        filenames.append("anchors_"+str(args.r)+"/A"+"{0:0=4d}".format(i))
    else:
        print("skipped A{0:0=4d}".format(i))

anchor_sets = []
for filename in filenames:
    anchor_sets.append(ProcessSeq.anchor_set_from_file(filename))

ProcessSeq.save_meta_data("anchors_"+str(args.r)+"/meta.txt", anchor_sets)
