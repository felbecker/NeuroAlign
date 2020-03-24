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

train_losses = []
node_rp_losses = []
col_rp_losses = []
rel_occ_losses = []
mem_logs_losses = []

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

    train_losses.append(train_loss_sum/config["batch_size"])
    node_rp_losses.append(l_node_rp_sum/config["batch_size"])
    col_rp_losses.append(l_col_rp_sum/config["batch_size"])
    rel_occ_losses.append(l_rel_occ_sum/config["batch_size"])
    mem_logs_losses.append(l_mem_logs_sum/config["batch_size"])

    print(i, " l=", sum(train_losses[-100:])/len(train_losses[-100:]),
                "l_n=", sum(node_rp_losses[-100:])/len(node_rp_losses[-100:]),
                "c_n=", sum(col_rp_losses[-100:])/len(col_rp_losses[-100:]),
                "l_mem_log=", sum(mem_logs_losses[-100:])/len(mem_logs_losses[-100:]),
                "l_rel_occ", sum(rel_occ_losses[-100:])/len(rel_occ_losses[-100:]))

    if i % config["savestate_milestones"] == 0 and i > 0:
        predictor.save()
