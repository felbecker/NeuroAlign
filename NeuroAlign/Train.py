import argparse
import MSA
import Model
import Trainer
from Config import config
import numpy as np

parser = argparse.ArgumentParser(description='Trains and tests a NeuroAlign model for the simple case of exact nucleotide matches.')
parser.add_argument("-n", type=int, default=5000, help="number of training examples")
parser.add_argument("-dir", type=str, default="./data_20", help="directory with data files")
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
    batch = []
    while len(batch) < config["batch_size"]:
        batch.append(msa[np.random.randint(len(msa))])
    train_loss_sum = 0
    l_node_rp_sum = 0
    l_col_rp_sum = 0
    l_rel_occ_sum = 0
    l_mem_logs_sum = 0
    for m in batch:
        n_rp, c_rp, rel_occ, mem_logits, train_loss, l_node_rp, l_col_rp, l_rel_occ, l_mem_logs = trainer.train(m)
        train_loss_sum += train_loss.numpy()
        l_node_rp_sum += l_node_rp.numpy()
        l_col_rp_sum += l_col_rp.numpy()
        l_rel_occ_sum += l_rel_occ.numpy()
        l_mem_logs_sum += l_mem_logs.numpy()

    print(i, " l=", train_loss_sum/config["batch_size"],
                "l_n=", l_node_rp_sum/config["batch_size"],
                "c_n=", l_col_rp_sum/config["batch_size"],
                "l_mem_log=", l_mem_logs_sum/config["batch_size"],
                "l_rel_occ", l_rel_occ_sum/config["batch_size"])

    if i % config["savestate_milestones"] == 0 and i > 0:
        predictor.save()
