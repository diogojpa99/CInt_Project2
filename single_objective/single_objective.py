from deap import base, tools, creator, algorithms
import time
import random
from math import sqrt, pow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


########### Case Studies ###########

cust_ord = pd.read_csv('CustOrd.csv')

# Centered
dists_cent = pd.read_csv('CustDist_WHCentral.csv')
xy_cent = pd.read_csv('CustXY_WHCentral.csv')

# Not Centered
dists_corn = pd.read_csv('CustDist_WHCorner.csv')
xy_corn = pd.read_csv('CustXY_WHCorner.csv')

# Number of costumers
n_costumers = 10
#n_costumers = 30
#n_costumers = 50

# Total number of products per 50 costumers
#print(sum(cust_ord['Orders'])) 

########### Functions ############

# Plot Costumer location
def plot_costumer_location(xy, max_client):
    
    plt.scatter(xy['X'][0:max_client], xy['Y'][0:max_client])
    plt.scatter(xy['X'][0], xy['Y'][0], c = '#d62728' , label = "Warehouse")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper right')
    plt.title('Costumer location')
    plt.show()
    
    return

# Cost fuction we want to minimize
# Hard restriction: Truck max capacity = 1000 products

# Funtion to save statistics across diferent generations
def SaveSatistics(individual):
    return individual.fitness.values

########### Initializations ############

# (1)
# Objective: Minimize a Cost Fuction
# i.e. Define the fitness: We want to find the least expensive path
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# (2)
# Individuals of the population - type: list
creator.create("Individual", list , fitness=creator.FitnessMin)

# (3)
# Obtain base (register everything)
toolbox = base.Toolbox()

# (4)
# Register Genes
# The genes will be a list of a possible path
# Were each index is a costumer
toolbox.register("Genes", np.random.permutation, n_costumers)

# (5)
# Register the individuals
toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.Genes) 

# (6)
# Register Population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# (7)
# Crossover operator
toolbox.register("mate", tools.cxUniform)

# (8)
# Mutation operator
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

# (9)
# Selection operator 
toolbox.register("select", tools.selTournament, tournsize=5)

# (10)
# Solution Evaluation
#toolbox.register("evaluate", f1)

# (11)
# Save statistics across genarations
stats = tools.Statistics(SaveSatistics)

# (12)
# Register different statistics
stats.register('mean', np.mean) 
stats.register('min', np.min)
stats.register('max', np.max)

# (13)
# Save Hall of Fame - Best individual
# Maximizer, or minimizer (I think)
hof = tools.HallOfFame(1)

# (14)
# Initialized the following probabilities
# CXPB  is the probability with which two individualsare crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.7, 0.2


########## main() ###########
def main():
    
    random.seed(64)
        
    # (15)
    # Initiate population
    pop = toolbox.population(n=100)
        
    start_time1 = time.process_time() # Program time
    
    # (16)
    # Run evolutionary algorithm
    result, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB,
                                      stats=stats, ngen=100, halloffame=hof, verbose=True)

    #print('Result:', result)
    print('Hall Of Fame:', hof)
    print ("Time Used ---> ", time.process_time() - start_time1, "seconds")

    return

'''if __name__ == "__main__":
    main()'''
