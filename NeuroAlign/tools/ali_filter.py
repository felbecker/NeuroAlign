import argparse

parser = argparse.ArgumentParser(description='Checks basic alignment stats like max and average lengths of a dataset.')
parser.add_argument("-n", type=int, default=200, help="number of files in source")
parser.add_argument("-t", type=int, default=250, help="threshold for alignment len")
parser.add_argument("-source_dir", type=str, default="./data_20_test/tcoffee", help="directory with data files")
parser.add_argument("-target_dir", type=str, default="./data_20_test/tcoffee", help="directory to put new alignment files")
args = parser.parse_args()

lens = []
hit = 1
for i in range(args.n):
    with open(args.source_dir+"/A"+"{0:0=4d}".format(i+1)+".fa", "r") as file:
        lines = file.readlines()
        if len(lines[1]) < args.t:
            with open(args.target_dir+"/A"+"{0:0=4d}".format(hit)+".fa", "w") as out_file:
                for line in lines:
                    out_file.write(line)
            hit += 1

print(hit, " hits")
