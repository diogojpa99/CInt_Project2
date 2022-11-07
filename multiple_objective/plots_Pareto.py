import numpy as np
import matplotlib.pyplot as plt


################### Imports ######################
MOP_10 = np.load('minStats_10.npy')
MOP_30 = np.load('minStats_30.npy')
MOP_50 = np.load('minStats_50.npy')


#################### Plots #######################


plt.plot(MOP_10[:,0], MOP_10[:,1], 'bo', label = '10 Customers')
plt.plot(MOP_30[:,0], MOP_30[:,1], 'go', label = '30 Customers')
plt.plot(MOP_50[:,0], MOP_50[:,1], 'ro', label = '50 Customers')


plt.xlabel('Cost')
plt.ylabel('Dist')
plt.legend(loc='lower right')
plt.title('Pareto Front values for 30 runs')
plt.show()

