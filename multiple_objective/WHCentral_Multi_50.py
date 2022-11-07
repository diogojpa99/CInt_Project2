#%%
from deap import base, tools, creator, algorithms
import time
import array
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deap import benchmarks
from deap.benchmarks.tools import hypervolume

########### Init ###########

# Orders
cust_ord = pd.read_csv('CustOrd.csv')

# Centered
dists_cent = pd.read_csv('CustDist_WHCentral.csv')
xy_cent = pd.read_csv('CustXY_WHCentral.csv')

# Number of costumers
n_costumers = 50

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
#%%
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
    
    We want to minimize the measure of the cost of transporting the products
    The cost will be higher when the truck is transporting a larger number of products
    '''
    individual = [x + 1 for x in individual]
    capacity = 1000    
    distances = []
    distances.append(dist[0,individual[0]]) # Distance between the warehouse and the first client
    
    travel_cost = []
    travel_cost.append(capacity * dist[0,individual[0]])

    for i in range (len(individual)-1):
        capacity -= cust_ord['Orders'][individual[i]]

        # Try to simulate the truck going to zero 
        if cust_ord['Orders'][individual[i+1]] > capacity or capacity == 0:
            distances.append(dist[individual[i],0]) # Truck has to go to from client i to warehouse
            travel_cost.append(capacity * dist[individual[i], 0])
            
            distances.append(dist[0,individual[i+1]])  # And then from the warehouse to client i+1
            capacity = 1000 # Full capacity again
            travel_cost.append(capacity * dist[0,individual[i+1]])
            #travel_cost=[]

        else: 
            distances.append(dist[individual[i], individual[i+1]])
            travel_cost.append(capacity * dist[individual[i],individual[i+1]])

    distances.append(dist[individual[int(len(individual)-1)],0]) # Return to warehouse
    travel_cost.append(capacity * dist[individual[int(len(individual)-1)],0])


    return sum(travel_cost), sum(distances),

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
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))

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
toolbox.register("mate", tools.cxOrdered)#tools.cxUniformPartialyMatched, indpb=0.09
#toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.05)

# (8)
# Mutation operator
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)

# (9)
# Selection operator 
toolbox.register("select", tools.selNSGA2) #tools.selNSGA2 #selSPEA2 #tools.selTournament, tournsize = 22) #assignCrowdingDist #sortNondominated

# (10)
# Solution Evaluation
toolbox.register("evaluate", Cost_Function)

# (11)
# Dealing with Constraints: Penalties
toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 1000, penalty_fxn)) 



########## main() ###########

def main(seed=None):

    # Run's Mean
    save_min_cost = []
    save_min_dist = []

    save_pareto_front_plot = []

    '''
    # 'BRUTE FORCE' for define the reference point in the hypervolume curve
    # Reference point = [worst cost, worst dist] = [max_cost, max_dist]
    # Worst cost: Highest cost achieved and similarly for distance
    
    max_cost = 600000 #.0 
    max_dist = 700 #1949.0
    # This values are achieved by running the program several times and
    # saving the values on worst_cost and worst_dist
    worst_cost = []
    worst_dist = []
    
    # Picking the best population
    # Best population: Minimize both distance and the cost (sum)
    
    worst_values =max_cost  + max_dist

    # Worst values = 2 since they are normalized between [0,1]
    We have proceded with normalization instead'''

    worst_values = 2

    for i in range(30):
    
        # Set a random seed
        random.seed(i+34)

        # Save statistics across genarations
        stats = tools.Statistics(SaveSatistics)
                
        # Register different statistics
        stats.register('mean', np.mean, axis=0) 
        stats.register('min', np.min, axis=0)
        stats.register('max', np.max, axis=0)

        # Initialized the following probabilities
        # CXPB  is the probability with which two individualsare crossed
        CXPB = 0.95
        
        # Initialize Pareto Front
        # Retrieve the best non dominated individuals of the evolution
        pareto_front = tools.ParetoFront() 
        pareto_front_plot = tools.ParetoFront()
                
        # Initialize Hypervolume
        HyperV = np.zeros(n_genarations)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "mean", "min", "max"
        
        
        # Initiate population
        pop = toolbox.population(n=n_population)
        
        # Evaluate the individuals with an invalid fitness - Atribute scores to the population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        #print(logbook.stream)

        
        # Begin the generational process
        for gen in range(1, n_genarations):
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            # Apply crossover and mutation on the offspring
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)
                
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            '''
            # MUTPB is the probability for mutating an individual: 
            MUTPB = 0.9
            for mutant in offspring:
            # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            '''
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, n_population)
            record = stats.compile(pop)
            #record = mstats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            #print(logbook.stream)

            # Calculating reference points for hypervolume
            if gen == 1:
                ref_max_cost = logbook[gen]['max'][0]
                ref_max_dist = logbook[gen]['max'][1]
                #print('ref_max_cost:')
                #print(ref_max_cost)

            # Calculating hypervolume for each generation
            HyperV[gen] = hypervolume(pop, [ref_max_cost, ref_max_dist])
            pareto_front_plot.update(pop)

            ''' BRUTE FORCE
            # This function calculate the reference point depending on the number of customers
            # Reference point can simply be found by constructing a vector of worst objective functions
            # In our case, worst objective functions implies bigger distance and bigger cost function
            worst_cost.append(record['max'][0])
            worst_dist.append(record['max'][1])
            '''

        save_pareto_front_plot.append([ind.fitness.values for ind in pareto_front_plot.items])
        ######################## PARETO FRONT AND HYPERVOLUME################
        # First we need to select the best population
        # We must compare the best pop from each run(30)
        # We need to normalize the objective functions so that both have the same decision power
        # The best population is the one that minimize both the cost and distance
    
        #  Normalize cost
        cost_max = ref_max_cost
        #cost_min = record['min'][0]
        cost_min = 0
        #cost_norm = (record['mean'][0] - cost_min) / (cost_max - cost_min)
        cost_norm = (record['min'][0] - cost_min) / (cost_max - cost_min)
        #print('cost_norm:', cost_norm)

        # Normalize dist
        dist_max = ref_max_dist
        #dist_min = record['min'][1]
        dist_min = 0
        #dist_norm = (record['mean'][1] - dist_min) / (dist_max - dist_min)
        dist_norm = (record['min'][1] - dist_min) / (dist_max - dist_min)
        
        # Threshold for selecting the best population
        sum_cost_dist = cost_norm + dist_norm
        

        if (sum_cost_dist) <  worst_values: 
            # If we have a pair of values (cost, dist) that 'minimizes' the sum, then:
            pop_best = pop
            # We need the hypervolume that correspond to the best population
            HyperV_best = HyperV
            # Update the values to catch a better population 
            worst_values = sum_cost_dist

            # Hypervolume for the best population
            HyperV_best = HyperV
        
        # Plot for information about each run
        #print(record)
        # Saving stats
        save_min_cost.append(record['min'][0])
        save_min_dist.append(record['min'][1])


     
    print('Mean cost:', np.mean(np.array(save_min_cost)))
    #print(np.shape(np.array(save_min_cost)))
    print('Mean dist:', np.mean(np.array(save_min_dist)))
    #print(np.shape(np.array(save_min_dist)))

    # Update the pareto front with the best chosen population
    pareto_front.update(pop_best)
    # Plot Pareto Front    
    front = np.array([ind.fitness.values for ind in pareto_front.items])
    
    
    plt.figure()
    plt.scatter(front[:,0], front[:,1], c="r")
    plt.title('Pareto Front for the best population')
    plt.axis("tight")
    plt.xlabel('Cost')
    plt.ylabel('Dist')
    plt.show()   
    

    # Indices that correspond to the minimum cost and dist values
    min_cost= np.argmin(front[:,0]) 
    min_dist = np.argmin(front[:,1])
    
    print('')
    print('Min Cost = (%s,%s)' % (front[min_cost,0],front[min_cost,1]))
    print('Min Dist = (%s,%s)' % (front[min_dist,0],front[min_dist,1]))
    print('')
    
    
    # Plot hypervolume 
    plt.figure()
    plt.plot(HyperV_best)
    plt.title('Hypervolume curve for the best population')
    plt.xlabel('#Generations')
    plt.ylabel('Hypervolume')
    plt.show()
    
    
    #print('Hypervolume evolution:')
    #print(HyperV_best)
    #print('')

    #print("Final population hypervolume is %f" % hypervolume(pop_best, [ref_max_cost, ref_max_dist])) # Utiliza apenas o valor final da ultima run
    #print('')

    ''' BRUTE FORCE
    print('Worst Cost:', np.max(np.array(worst_cost)))
    print('Worst dist:', np.max(np.array(worst_dist)))
    '''
    

    # Save multiple paretos for plot
    plot_pareto = []
    for i in save_pareto_front_plot:
        plot_pareto += i
    
    np.save('minStats_50.npy', plot_pareto)
    np.save('Hypervolume_50.npy', HyperV_best)

    return pop, logbook, record

if __name__ == "__main__":
    
    pop, stats, record = main()


#print('Stats:')
#print(stats)