from deap import base, tools, creator, algorithms
import time
import random
from math import sqrt, pow, sin
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
#n_costumers = 30 + 1 #Truck has to go to the warehouse once
#n_costumers = 50 + 2 #Truck has to go to the warehouse twice

# Total number of products per 50 costumers
#print(sum(cust_ord['Orders'])) 

# Number of genarations
n_genarations = 250

# Max number of the population
n_population = 40

if (n_population*n_genarations) > 100000:
    print('ERROR: Maximum number of evaluations has exceeded')
    exit(0)
    
# Dist_cent 'preprocessing'
dist = dists_cent.to_numpy()
dist= np.delete(dist, 0, axis=1)

# Dist_corn 'preprocessing'
'''dist = dists_corn.to_numpy()
dist= np.delete(dist, 0, axis=1)'''

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
def Cost_Function(individual):
    
    distances = []
    distances.append(dist[0,individual[0]]) # Distance between the warehouse and the first client
    
    for i in range (len(individual)-1):
        distances.append(dist[individual[i], individual[i+1]]) # Distance between each costumer in our possible solution
    
    return sum(distances),

def check_feasiblity(individual):
    '''
    Feasibility function for the individual. 
    Returns True if individual is feasible (or constraint not violated),
    False otherwise
    '''
    if (n_costumers==10 and (0 in individual)) or (len(set(individual)) != len(individual)):
        # Indiviual contains repeated values
        return True
    else:
        return False


def penalty_fxn(individual):
    '''
    Penalty function to be implemented if individual is not feasible or violates constraint
    It is assumed that if the output of this function is added to the objective function fitness values,
    the individual has violated the constraint.
    '''
    return Cost_Function(individual=individual)**2

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
toolbox.register("mate", tools.cxPartialyMatched)

# (8)
# Mutation operator
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

# (9)
# Selection operator 
toolbox.register("select", tools.selTournament, tournsize=5)

# (10)
# Solution Evaluation
toolbox.register("evaluate", Cost_Function)

# (11)
# Constraints
toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 1e30)) 

# (12)
# Save statistics across genarations
stats = tools.Statistics(SaveSatistics)

# (13)
# Register different statistics
stats.register('mean', np.mean) 
stats.register('min', np.min)
stats.register('max', np.max)

# (14)
# Save Hall of Fame - Best individual
# Maximizer, or minimizer (I think)
hof = tools.HallOfFame(1)

# (15)
# Initialized the following probabilities
# CXPB  is the probability with which two individualsare crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.7, 0.2


########## main() ###########
def main():
    
    random.seed(64)
        
    # (16)
    # Initiate population
    pop = toolbox.population(n=40)
        
    start_time1 = time.process_time() # Program time
    
    # (17)
    # Run evolutionary algorithm
    result, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB,
                                      stats=stats, ngen=30, halloffame=hof, verbose=True)

    #print('Result:', result)
    print('Hall Of Fame:', hof)
    print ("Time Used ---> ", time.process_time() - start_time1, "seconds")

    return

if __name__ == "__main__":
    main()
