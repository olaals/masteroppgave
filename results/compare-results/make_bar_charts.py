import matplotlib.pyplot as plt
import numpy as np


# pbr test set
fig,ax = plt.subplots()
plt.xticks(rotation=45)
labels = ['End to end machine learning', 'Geometric consistency', 'Epipolar consistency', 'Epipolar and geoemtric consistency', 'Cross-section geometry assumption']
x = np.arange(len(labels))
ax.set_xticks(x)
ax.set_xticklabels(labels)


plt.yscale('log')
plt.bar(0, 2, color='#dd4444')
plt.bar(1, 100, color='#55cc55')
plt.bar(2, 30, color='c')
plt.bar(3, 10, color='r')
plt.bar(4, 1, color='b')
plt.savefig("sdffsd.png")
plt.show()
