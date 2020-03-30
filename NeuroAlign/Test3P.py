import argparse
import MSA

from Config import config

parser = argparse.ArgumentParser(description='Tests the latest NeuroAlign model.')
parser.add_argument("-n", type=int, default=200, help="number of testing examples")
parser.add_argument("-dir_3p", type=str, default="./data_20_test/tcoffee", help="directory with 3p data files")
parser.add_argument("-dir_ref", type=str, default="./data_20_test", help="directory with reference data files")
args = parser.parse_args()

#load the training dataset
def load(dir):
    msa = []
    for i in range(1,args.n+1):
        filepath = dir + "/A"+"{0:0=4d}".format(i)+".fa"
        msa.append(MSA.Instance(filepath))
    return msa

ref = load(args.dir_ref)
thirdp = load(args.dir_3p)

ps = 0
rs = 0
for ref_m, thirdp_m in zip(ref, thirdp):
    p,r = ref_m.recall_prec(thirdp_m.membership_targets)
    ps += p
    rs += r

print("precision=", ps/len(ref), "recall=", rs/len(ref))
