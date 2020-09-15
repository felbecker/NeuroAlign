import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Thins out a fasta file, producing a new file with only xx % of the sequences.')
parser.add_argument("-f", type=str, default="", help="fasta file")
parser.add_argument("-thin", type=np.float32, default="", help="thinning probability")
parser.add_argument("-out", type=str, default="", help="output file")
args = parser.parse_args()


with open(args.f) as f:
    content = f.readlines()

with open(args.out, "w") as out:
    write = False
    for i,line in enumerate(content):
        if len(line)>0:
            if line[0]=='>':
                write = np.random.rand() < args.thin
                if write:
                    out.write(line)
            else:
                if write:
                    out.write(line)
