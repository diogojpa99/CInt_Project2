import numpy as np
import matplotlib.pyplot as plt


################### Imports ######################

SOP10Cent50 = np.load('30-Costumers/stats/WHCentral_Ord50best.npy')
SOP10CentOrd = np.load('30-Costumers/stats/WHCentral_OrdFilebest.npy')
SOP10Corn50 = np.load('30-Costumers/stats/WHCorner_Ord50best.npy')
SOP10CornOrd = np.load('30-Costumers/stats/WHCorner_OrdFilebest.npy')

#################### Plots #######################

plt.plot(SOP10Cent50, label = 'WHCentral_Ord50')
plt.plot(SOP10CentOrd, label = 'WHCentral_OrdFile')
plt.plot(SOP10Corn50, label = 'WHCorner_Ord50')
plt.plot(SOP10CornOrd, label = 'WHCorner_OrdFile')
plt.xlabel('#Generations')
plt.ylabel('Total Distance for the best run')
plt.legend(loc='upper right')
plt.title('Convergence Curve for 30 costumers')
plt.show()