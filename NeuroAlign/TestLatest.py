import argparse
import MSA
import Model

from Config import config

parser = argparse.ArgumentParser(description='Tests the latest NeuroAlign model.')
parser.add_argument("-n", type=int, default=200, help="number of testing examples")
parser.add_argument("-dir", type=str, default="./data_50_test", help="directory with data files")
args = parser.parse_args()

#load the training dataset
msa = []
for i in range(1,args.n+1):
    filepath = args.dir + "/A"+"{0:0=4d}".format(i)+".fa"
    msa.append(MSA.Instance(filepath))

#instantiate the predictor
predictor = Model.NeuroAlignPredictor(config, msa[0])
predictor.load_latest()

ps = 0
rs = 0
for m in msa:
    _,_,mem = predictor.predict(m, m.alignment_len)
    p,r = m.recall_prec(mem)
    ps += p
    rs += r

print("precision=", ps/len(msa), "recall=", rs/len(msa))
