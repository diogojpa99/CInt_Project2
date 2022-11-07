import numpy as np
import matplotlib.pyplot as plt


################### Imports ######################
MOP_10 = np.load('Hypervolume_10.npy')
MOP_30 = np.load('Hypervolume_30.npy')
MOP_50 = np.load('Hypervolume_50.npy')


#################### Plots #######################


plt.plot(MOP_10, 'b', label = '10 Customers')
plt.plot(MOP_30, 'g', label = '30 Customers')
plt.plot(MOP_50, 'r', label = '50 Customers')


plt.xlabel('Generations')
plt.ylabel('Hypervolume for 30 runs')
plt.legend(loc='lower right')
plt.title('Pareto Front values for 30 runs')
plt.show()

