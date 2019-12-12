import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

names = []
scores = []
with open("scores", "r") as f:
    for line in f:
        parts = line.strip().split(' ')
        names.append(parts[0])
        scores.append(np.float32(parts[1]))

y_pos = np.arange(len(names))

barlist = plt.bar(y_pos, scores, align='center', alpha=0.5)
plt.xticks(y_pos, names)
barlist[0].set_color('r')
barlist[1].set_color('g')
barlist[2].set_color('y')
barlist[3].set_color('m')
plt.ylabel('Jaccard-Index')
plt.ylim((0.8, 1))  
plt.title('Comparison with a test set of reference alignments')

plt.show()
