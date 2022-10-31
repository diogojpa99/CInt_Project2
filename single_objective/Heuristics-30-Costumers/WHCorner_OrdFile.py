from deap import base, tools, creator, algorithms
import time
import random
from math import pow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


########### Init ###########

# Orders
cust_ord = pd.read_csv('CustOrd.csv')

# Not Centered
dists_corn = pd.read_csv('CustDist_WHCorner.csv')
xy_corn = pd.read_csv('CustXY_WHCorner.csv')

# Number of costumers
n_customers = 30 

# Number of genarations
n_genarations = 100

# Max number of the population
n_population = 100

if (n_population*n_genarations) > 100000:
    print('ERROR: Maximum number of evaluations has exceeded')
    exit(0)
    
# Dist_corn 'preprocessing'
dist = dists_corn.to_numpy()
dist= np.delete(dist, 0, axis=1)

########### Functions ############

# Plot Costumer location
def plot_costumer_location_corn(xy, max_client):
    
    fig, ax = plt.subplots()
    ax.scatter(xy['x'][0:max_client],  xy['y'][0:max_client])
    ax.scatter(xy['y'][0], xy['y'][0], c = '#d62728' , label = "Warehouse")
    
    for i, txt in enumerate(xy['Customer XY'][0:max_client]):
        ax.annotate(txt, (xy['x'][i], xy['y'][i]))
    
    plt.show()
        
    return

# Cost fuction we want to minimize
# Hard restriction: Truck max capacity = 1000 products    
def Cost_Function(individual):

    individual = [x + 1 for x in individual]
    capacity = 1000    
    distances = []
    distances.append(dist[0,individual[0]]) # Distance between the warehouse and the first client
    
    for i in range (len(individual)-1):
        capacity -= cust_ord['Orders'][individual[i]]
        # Try to simulate the truck going to zero 
        if cust_ord['Orders'][individual[i+1]] > capacity or capacity == 0:
            distances.append(dist[individual[i],individual[0]]) # Truck has to go to from client i to warehouse
            distances.append(dist[0,individual[i+1]])  # And then from the ware house to client i+1
            capacity = 1000 # Full capacity again
        else: distances.append(dist[individual[i], individual[i+1]]) 

    distances.append(dist[individual[int(len(individual)-1)],0])
    
    return sum(distances),

def check_feasiblity(individual):
    '''
    Feasibility function for the individual. 
    Returns True if individual is feasible (or constraint not violated),
    False otherwise
    '''
    # Indiviual contains repeated values
    if (len(set(individual)) != len(individual)): return False
    else: return True


def penalty_fxn(individual):
    '''
    Penalty function to be implemented if individual is not feasible or violates constraint
    It is assumed that if the output of this function is added to the objective function fitness values,
    the individual has violated the constraint.
    '''
    return pow(int(Cost_Function(individual=individual)[0]),2)

def SaveSatistics(individual):
    '''
    Funtion that saves statistics across diferent generations
    '''
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
toolbox.register("Genes", np.random.permutation, n_customers)

# (5)
# Register the individuals
toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.Genes) 

# (6)
# Register Population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# (7)
# Crossover operator
toolbox.register("mate", tools.cxOrdered)

# (8)
# Mutation operator
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# (9)
# Selection operator 
toolbox.register("select", tools.selTournament, tournsize = 15)

# (10)
# Solution Evaluation
toolbox.register("evaluate", Cost_Function)

# (11)
# Dealing with Constraints: Penalties
toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 1000, penalty_fxn)) 

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
CXPB, MUTPB = 0.8, 0.8

########## main() ###########
def main():
    
    min_array = []
    short_dist = 100000
    best_run = np.empty(n_genarations,)
    
    for i in range (30):
        
        random.seed(i+34)
            
        # (16)
        # Initiate population
        pop = toolbox.population(n=n_population)
            
        start_time1 = time.process_time() # Program time
        
        # (17)
        # Run evolutionary algorithm
        result, log = algorithms.eaSimple(population=pop, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB,
                                        stats=stats, ngen=n_genarations, halloffame=hof, verbose=False)
        
        # Saving stats
        min_array.append(log[n_genarations]['min'])
        if log[n_genarations]['min'] < short_dist:
            for j in range (n_genarations): 
                best_run[j]=log[j]['min']
            short_dist = log[n_genarations]['min']
            
        
        real_hof = [x + 1 for x in hof[0]]
        #print('Hall Of Fame:',real_hof)
        #print ("Time Used ---> ", time.process_time() - start_time1, "seconds")
        
    print('MEAN:', np.mean(min_array))
    print('STD:', np.std(min_array))
    np.save('30-Costumers/stats/WHCorner_OrdFilebest.npy', best_run)
    
    return

if __name__ == "__main__":
    main()
