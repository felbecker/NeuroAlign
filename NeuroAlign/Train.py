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
alphabet = ['A', 'C', 'G', 'T'] if config["type"] == "nucleotide" else ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X']
for i in range(1,args.n+1):
    filepath = args.dir + "/A"+"{0:0=4d}".format(i)+".fa"
    #print(filepath)
    #print("Reading ", filepath)
    inst = MSA.Instance(filepath, alphabet)
    if inst.valid:
        msa.append(inst)

#instantiate the predictor and the trainer
predictor = Model.NeuroAlignPredictor(config, msa[0])
predictor.load_latest()
trainer = Trainer.NeuroAlignTrainer(config, predictor)

train_losses = []

for i in range(config["num_training_iteration"]):
    batch = []
    while len(batch) < config["batch_size"]:
        batch.append(msa[np.random.randint(len(msa))])
    train_loss_sum = 0
    for m in batch:
        mem, train_loss, l_mem = trainer.train(m)
        train_loss_sum += train_loss.numpy()

    train_losses.append(train_loss_sum/config["batch_size"])

    print(i, " l=", sum(train_losses[-100:])/len(train_losses[-100:]))

    if i % config["savestate_milestones"] == 0 and i > 0:
        predictor.save()
