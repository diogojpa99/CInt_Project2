from deap import base, tools, creator, algorithms
import time
import random
from math import pow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial


########### Init ###########

# Orders
cust_ord = pd.read_csv('CustOrd.csv')

# Centered
dists_cent = pd.read_csv('CustDist_WHCentral.csv')
xy_cent = pd.read_csv('CustXY_WHCentral.csv')

# Number of costumers
n_customers = 50

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

########### Functions ############

def plot_costumer_location_cent(xy, max_client):
    '''
    Plot Customer location 
    ''' 
    fig, ax = plt.subplots()
    ax.scatter(xy['X'][0:max_client],  xy['Y'][0:max_client])
    ax.scatter(xy['X'][0], xy['Y'][0], c = '#d62728' , label = "Warehouse")
    
    for i, txt in enumerate(xy['Customer XY'][0:max_client]):
        ax.annotate(txt, (xy['X'][i], xy['Y'][i]))
    
    plt.show()
        
    return
 
def Cost_Function(individual):
    '''
    Cost fuction we want to minimize
    We want to minimize the sum of the distances traveled
    We have to take into account the capacity of the truck
    If the capacity is surpassed then the truck has to return to the warehouse
    The truck can only visit a customer once
    '''
    individual = [x + 1 for x in individual]
    capacity = 1000    
    distances = []
    distances.append(dist[0,individual[0]]) # Distance between the warehouse and the first client
    
    for i in range (len(individual)-1):
        capacity -= cust_ord['Orders'][individual[i]]
        # Try to simulate the truck going to zero 
        if cust_ord['Orders'][individual[i+1]] > capacity or capacity == 0:
            distances.append(dist[individual[i],0]) # Truck has to go to from client i to warehouse
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

def Heuristics_individual_cent(n_customer):
    '''
    Create the individual using Heuristics
    '''
    indiv = []
    xy_cust = xy_cent[xy_cent['Customer XY'] < n_customer+1]
    xy_cust = xy_cust.drop(0)
        
    # (1)
    # Spilt the x axis in half
    #horiz_mid = (max(xy['X'])+min(xy['X']))/2
    horiz_mid = 50
    
    # (2)
    # Split the population in two: left and right
    left_cust = xy_cust[xy_cust['X'] < horiz_mid]
    right_cust = xy_cust[xy_cust['X'] >= horiz_mid]
    
    # (3)
    # Start with the customers from the left side: Down -> Up
    left_cust = left_cust.sort_values(by=['Y'])
    indiv = left_cust['Customer XY'].values

    # (4)
    # Customers in the right side: Up -> Down
    right_cust = right_cust.sort_values(by=['Y'], ascending=False)
    indiv = np.concatenate((indiv,right_cust['Customer XY'].values), axis = None)
    
    indiv = [x - 1 for x in indiv]
    
    return indiv

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
# Register Heuristic Gene
toolbox.register("Gene_heuristic",  partial(Heuristics_individual_cent,n_customers))

# (5)
# Register the individuals
toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.Genes) 
# Register Heuristic Individual
toolbox.register("individual_heuristic", tools.initIterate, creator.Individual, toolbox.Gene_heuristic)

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
toolbox.register("select", tools.selTournament, tournsize = 22)

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
        #Include heuristics individual in population
        heuris_indv = toolbox.individual_heuristic()
        pop[0] = heuris_indv 
            
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
    print('Heuristics Path:',  [x + 1 for x in heuris_indv], '| Distance: ', Cost_Function(heuris_indv)[0])
    np.save('Heuristics-50-Costumers/stats/WHCentral_OrdFilebest.npy', best_run)
    
    return

if __name__ == "__main__":
    main()