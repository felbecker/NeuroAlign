import argparse
import os

parser = argparse.ArgumentParser(description='Converts xml 2 fasta')
parser.add_argument("-f", type=str, default="", help="xml file name")
args = parser.parse_args()

with open(args.f) as f:
    content = f.readlines()
    seq = []
    names = []
    for line in content:
        if "<seq-name>" in line:
            line = line[10:]
            names.append(line[:(len(line)-12)])
        elif "<seq-data>" in line:
            seq.append(line[10:].strip())

with open(os.path.splitext(args.f)[0]+".fasta", "w") as f:
    for name, seq in zip(names, seq):
        f.write(">"+name+"\n")
        f.write(seq+"\n")


