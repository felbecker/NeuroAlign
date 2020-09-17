import tensorflow as tf
import numpy as np
import MSA
import SequenceModel


##################################################################################################
##################################################################################################

#load the data
msa = MSA.Instance("Pfam_very_thin.fasta", SequenceModel.ALPHABET, gaps=False)

if not msa.valid:
    print("Invalid data.")
    quit()

np.random.seed(0)

indices = np.array(range(len(msa.raw_seq)))
np.random.shuffle(indices)
train, test = np.split(indices, [int(len(msa.raw_seq)*(1-SequenceModel.VALIDATION_SPLIT))])


##################################################################################################
##################################################################################################

#provides batches of protein sequences
class SequenceBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, split, train):
        self.split = split
        self.train = train

    def __len__(self):
        if self.train:
            if len(self.split) > 10*SequenceModel.BATCH_SIZE:
                return int(np.floor(len(self.split) / (10*SequenceModel.BATCH_SIZE)))
        return int(np.floor(len(self.split) / SequenceModel.BATCH_SIZE)) #steps per epoch

    def __getitem__(self, index):
        batch_indices = np.random.choice(self.split, size=SequenceModel.BATCH_SIZE)
        batch = [msa.raw_seq[i] for i in batch_indices]
        batch_lens = [s.shape[0] for s in batch]
        num_steps = max(batch_lens)
        seq = np.zeros((SequenceModel.BATCH_SIZE, num_steps, len(SequenceModel.ALPHABET)), dtype=np.float32)
        for j,(l,s) in enumerate(zip(batch_lens, batch)):
            seq[j,np.arange(l),s] = 1
        return seq, seq


train_gen = SequenceBatchGenerator(train, True)
val_gen = SequenceBatchGenerator(test, False)


##################################################################################################
##################################################################################################

model = SequenceModel.make_model()

data = val_gen.__getitem__(0)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=SequenceModel.CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_gen,
            validation_data=val_gen,
            epochs = SequenceModel.NUM_EPOCHS,
            verbose = 1,
            callbacks=[cp_callback])
