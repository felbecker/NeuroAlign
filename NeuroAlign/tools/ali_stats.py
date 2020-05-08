import argparse

parser = argparse.ArgumentParser(description='Checks basic alignment stats like max and average lengths of a dataset.')
parser.add_argument("-n", type=int, default=200, help="number of files")
parser.add_argument("-dir", type=str, default="./data_20_test/tcoffee", help="directory with data files")
args = parser.parse_args()

lens = []
for i in range(args.n):
    with open(args.dir+"/A"+"{0:0=4d}".format(i+1)+".fa", "r") as file:
        lines = file.readlines()
        lens.append(len(lines[1]))

print("Average alignment len: ", sum(lens)/len(lens))
print("Min alignment len: ", min(lens))
print("Max alignment len: ", max(lens))
