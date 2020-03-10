import argparse
import random
import MSA
import Model
import Trainer
from Config import config

parser = argparse.ArgumentParser(description='Trains and tests a NeuroAlign model for the simple case of exact nucleotide matches.')
parser.add_argument("-n", type=int, default=5000, help="number of training examples")
parser.add_argument("-dir", type=str, default="./data_50", help="directory with data files")
args = parser.parse_args()

#load the training dataset
msa = []
for i in range(1,args.n+1):
    filepath = args.dir + "/A"+"{0:0=4d}".format(i)+".fa"
    msa.append(MSA.Instance(filepath))

#instantiate the predictor and the trainer
predictor = Model.NeuroAlignPredictor(config, msa[0])
predictor.load_latest()
trainer = Trainer.NeuroAlignTrainer(config, predictor)

for i in range(config["num_training_iteration"]):
    random.shuffle(msa)
    train_loss_sum = 0
    l_node_rp_sum = 0
    l_mem_sum = 0
    for m in msa:
        n_rp, c_rp, mem, train_loss, l_node_rp, l_col_rp, l_mem = trainer.train(m)
        train_loss_sum += train_loss.numpy()
        l_node_rp_sum += l_node_rp.numpy()
        l_mem_sum += l_mem.numpy()

    print(i, " l=", train_loss_sum/len(msa), "l_n=", l_node_rp_sum/len(msa), "l_mem=", l_mem_sum/len(msa))

    if i % config["savestate_milestones"] == 0 and i > 0:
        predictor.save()
