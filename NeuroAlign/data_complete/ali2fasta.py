import argparse
import os

parser = argparse.ArgumentParser(description='Converts ali 2 fasta')
parser.add_argument("-f", type=str, default="", help="ali file name")
args = parser.parse_args()

with open(args.f) as f:
    content = f.readlines()
    seq = []
    names = []
    seq_reading = False
    skipped = False
    for line in content:
        if not seq_reading:
            if '>' in line: #opener
                names.append(line[1:].strip())
                seq_reading = True
                skipped = False
                seq.append("")
        elif not skipped:
            skipped = True
        else:
            if '*' in line:
                line = line.strip()
                seq[-1] += line[:len(line)-1]
                seq_reading = False
            else:
                seq[-1] += line.strip()

with open(os.path.splitext(args.f)[0]+".fasta", "w") as f:
    for name, seq in zip(names, seq):
        f.write(">"+name+"\n")
        f.write(seq+"\n")
