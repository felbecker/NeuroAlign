import numpy as np
import matplotlib.pyplot as plt

train_losses = []
test_losses = []

i = 0
with open("train_out", "r") as f:
    for line in f:
        parts = line.strip().split(' ')
        if parts[0]=='Iteration' and len(parts) > 11:
            train_losses.append((i, np.float32(parts[13])))
            i += 1
        elif parts[0]=='Test':
            test_losses.append((i, np.float32(parts[3])))

train_losses = np.matrix(train_losses)
test_losses = np.matrix(test_losses)

plt.plot(train_losses[:,0],train_losses[:,1],'r')
plt.plot(test_losses[:,0],test_losses[:,1],'b')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.title('NeuroAlign training and test loss')

plt.show()
