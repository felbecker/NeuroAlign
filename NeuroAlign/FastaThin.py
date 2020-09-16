import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Thins out a fasta file, producing a new file with only xx % of the sequences.')
parser.add_argument("-f", type=str, default="", help="fasta file")
parser.add_argument("-thin", type=np.float32, default="", help="thinning probability")
parser.add_argument("-minlen", type=int, default=0, help="discards all sequences below this length")
parser.add_argument("-maxlen", type=int, default=100000, help="discards all sequences above this length")
parser.add_argument("-out", type=str, default="", help="output file")
args = parser.parse_args()


with open(args.f) as f:
    content = f.readlines()

def write_batch(out, batch):
    l = sum([len(l) for l in batch[1:]])
    if l > args.minlen and l < args.maxlen:
        out.write("".join(batch))

with open(args.out, "w") as out:
    write = False
    for i,line in enumerate(content):
        if len(line)>0:
            if line[0]=='>': #header line
                if write:
                    write_batch(out, batch)
                batch = []
                write = np.random.rand() < args.thin
                batch.append(line)
            else: #take into account that the sequence may occupy multiple lines in the file
                batch.append(line)
                if i == len(content)-1 and write:
                    write_batch(out, batch)
