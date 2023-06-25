# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:50:09 2020

@author: maryam
"""

'''This code is used to implement the genetic algorithm on the quadratic assignment problem. 
It uses a permutation crossover operator, uniform crossover operator, and bit flip mutation. 
It uses tournament selection and sets a population size of 10. 
The stopping criteria is that the algorithm will stop after n iterations or if the fitness value is y'''

# Quadratic Assinment problem- 
#The objective is to minimze the flow cost between departments 
#flow cost  = flow * distance

#import libraries 
import numpy as np
import pandas as pd
import time
import random 
np.random.seed(777)

# GA parameters; pc, pm, n, g_max
# define pc; the probability of crossover 
pc = 0.7

# define pm; the probability of mutation 
pm = 0.2

# define n; the population size 
n = 50
#so i is 0 to n 
# define g_max; the max number of generations 
g_max = 250

chromosome_length = 12

#%%
# import distance matrix from excel
# remove first column and row indices 
dist_flow_mat = pd.read_csv('distflow.csv', header = None)

#%% convert to numpy
distflow_mat = dist_flow_mat.to_numpy()


#%% mirror the upper triangle of the matrix to create distance matrix
#a = np.matrix([[11,19, 6],[10,5,7], [3, 2, 4]])
#dist_mat = dist_mat + dist_mat.T - np.diag(np.diag(dist_mat))
#print(dist_mat)

#get the upper triangular part of this matrix
distflow_uppertr = distflow_mat[np.triu_indices(distflow_mat.shape[0], k = 0)]
# [1 2 3 5 6 9]

# put it back into a 2D symmetric array
size_dist= 12
distflow_mat = np.zeros((size_dist,size_dist))
distflow_mat[np.triu_indices(distflow_mat.shape[0], k = 0)]= distflow_uppertr 
dist_mat = distflow_mat + distflow_mat.T - np.diag(np.diag(distflow_mat))
#array([[1., 2., 3.],
#       [2., 5., 6.],
#       [3., 6., 9.]])

#%%  mirror the lower triangle of the matrix to create flow matrix
distflow_matf = dist_flow_mat.to_numpy()
distflow_lowertr = np.tril(distflow_matf ,k=-1)

size_flow = 12
#z = np.zeros((size_flow,size_flow))

flow_mat = distflow_lowertr + distflow_lowertr.T - np.diag(np.diag(distflow_lowertr))

#%% define columns
flow_matrix = pd.DataFrame(flow_mat)

flow_matrix = flow_matrix.rename(index={0: "D1", 1: "D2", 2: "D3", 3: "D4",
                                          4: "D5", 5: "D6", 6: "D7", 7: "D8",
                                          8: "D9", 9: "D10", 10: "D11", 
                                          11: "D12"})

flow_matrix.columns = ['D1','D2','D3','D4','D5', 'D6','D7', 'D8', 'D9','D10', 'D11','D12']


#%%define columns
dist_matrix =  pd.DataFrame(dist_mat)

dist_matrix = dist_matrix.rename(index={0: "D1", 1: "D2", 2: "D3", 3: "D4",
                                          4: "D5", 5: "D6", 6: "D7", 7: "D8",
                                          8: "D9", 9: "D10", 10: "D11", 
                                          11: "D12"})
dist_matrix.columns = ["D1",'D2','D3','D4','D5', 'D6','D7', 'D8', 'D9','D10', 'D11','D12']

#%% this function creates a random population with defined length
def initialize(chromosome):
    
    #dist_matrix = dist_matrix.to_numpy()
    int_solution =  random.sample((chromosome), len(chromosome))
#    new_flow = flow_matrix.reindex (index = int_solution, columns = int_solution )
#    new_flow = new_flow.to_numpy()
#    dist_mat = dist_matrix.to_numpy()
#    return new_flow, dist_mat
    return int_solution

#%% selection
def selection (chromosome, n):

#    parent0 = []
#    parent1 = []
    
    lists = np.empty((0, len(chromosome)))


        
    dic_min = {}
    #solution = {}
#    tup_min_0 = []
#    tup_min_1 = []
  #  min_value ={}
   # sum_cost = 0
    selected_parents={}  


    for p in range (2): 
        
        for i in range (n):
            random_chromosome = random.sample(chromosome, len(chromosome))
            x = np.reshape(random_chromosome, (1, len(chromosome)))
            lists= np.concatenate((lists, x), axis =0)
            
        #print('********************************************')
        #selected_parents={}  
        k = []
        # Set a length of the list to k
        dic_min[p] = {}
        
        for i in range(0, 3):
            # any random numbers from 0 to n
            list_name = range(10)
            k = random.sample(list_name, 3)

    #min_z = min(solution.keys(), key=(lambda i: solution[i]))    
     #   counter = 0 
        #print(k)
     #   row = 0 
        for i in k:
            #print(i, k)
            selected_parents[i] = lists[np.random.randint(0,len(lists))]
            #print('=================================')
            #print(selected_parents[i])
          
            new_flow = flow_matrix.reindex (index = selected_parents[i], columns = selected_parents[i])
            new_flow = new_flow.to_numpy()
#            print(new_flow)
            time.sleep(0)
            
            dist_mat = dist_matrix.to_numpy()
            new_flowcost = (new_flow * dist_mat)
            sum_flowcost = new_flowcost.sum()
            #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            #print(sum_flowcost, row)
            
            dic_min[p][i] = sum_flowcost
            #
           # print(dic_min[0],dic_min[1], '=======================,')

# parent_0
           
           # minkey_0 =  min(dic_min[0].keys(), key=(lambda k: dic_min[0][k]))
            #print(minkey_0)

            
    return selected_parents[min(dic_min[0],key = dic_min[0].get)], selected_parents[min(dic_min[1],key = dic_min[1].get)]            
#    return selected_parents[min(dic_min[0],key = dic_min[0].get)], selected_parents[min(dic_min[1],key = dic_min[1].get)]


#%% crossover function
def crossover(parent_0, parent_1):
    
    fp = np.random.randint(0,len(chromosome))
    
    if type(parent_0) != list: 
        parent_0_l = parent_0.tolist()
        l_tail_0 =(parent_0[:fp]).tolist()
    
    else:
        parent_0_l = parent_0
        l_tail_0 =(parent_0[:fp])
    
    if type(parent_1) != list: 
        parent_1_l = parent_1.tolist()
        l_tail_1 =(parent_1[:fp]).tolist()
    else:
        parent_1_l = parent_1
        l_tail_1 =(parent_1[:fp])
    #r_tail_0 =(parent_0[fp:]).tolist()
    
    
    #r_tail_1 =(parent_1[fp:]).tolist()

    list_diff_0 = [x for x in parent_1_l if x not in l_tail_0]
    # Run Test
    l_tail_0.extend((list_diff_0))
    child_0 = l_tail_0
    
    
    list_diff_1 = [x for x in parent_0_l if x not in l_tail_1]
    l_tail_1.extend((list_diff_1))
    child_1 = l_tail_1
    
    return child_0, child_1


#%% mutation - inversion 

def inversion_mutation (child): 
    
        rand_1= random.randint(0, len(child) - 1)
        rand_2 = random.randint(rand_1, len(child) - 1)
        if rand_1 == rand_2 :
            rand_2 = random.randint(rand_1, len(child) - 1)
        
        middle_part = child[rand_1:rand_2]
        inversed_middle = middle_part[::-1]
    
        child[:] = child[0:rand_1] + inversed_middle + child [rand_2:]
        mutated_child = child[:]
        return mutated_child

#%% mutation function
def mutation_cost(chromosome):
    
    new_flow = flow_matrix.reindex (index = chromosome, columns = chromosome )
    new_flow = new_flow.to_numpy()
    dist_mat = dist_matrix.to_numpy()
    new_flowcost = (new_flow*dist_mat)
    
    sum_flowcost = (new_flowcost.sum())
    return sum_flowcost


   #%%   Genetic algorithm for QAP
def ga_qap(n,g_max,pm, pc, int_solution):
#
    min_fitness = {}    
   # list_fitness = []
    min_list_cost = []

    
#    pc = 0.99
#    # define pm; the probability of mutation 
#    pm = 0.7
#    # define n; the population size 
#    n = 10
#    #so i is 0 to n 
#    # define g_max; the max number of generations 
#    g_max = 25 
    #chromosome_length = 12
    
    test_dict ={}
    
    for g in range(1, g_max+1):  
        test_dict[g] ={}
        min_fitness[g] = {}
        for i in range (1, n//2):
#            print('CAPITAL N', g, n)

            if g == 1 and i == 1:                
                parent_0, parent_1 = selection (int_solution, n//2)
#                
            else:
#                print('========>', i , mutated_child_0, mutated_child_1)
#                parent_0 = mutated_child_0
#                parent_1 = mutated_child_1
            
                rand_cross = np.random.random_sample()
            #
#            print('L2 ^^^^^^^^^^^^^^^^^^^^^^^^^', rand_cross,g, i, rand_cross < 1)
#
#            time.sleep(0)
                if (rand_cross < 1):
                    child_0, child_1 = crossover(parent_0, parent_1)
                    rand_mut = np.random.random_sample()
    #                print('L21 ----------------------------------', rand_mut, pm, rand_mut < pm)
    #                time.sleep(0)
                    if (rand_mut < pm):
                        mutated_child_0 = inversion_mutation(child_0)
                        mutated_child_1 = inversion_mutation(child_1)
                        
    
                    else:
                        mutated_child_0 = parent_0 
                        mutated_child_1 = parent_1
#            
#            print('======================================')
#            print(mutated_child_0)
#            print(mutated_child_1)
#            print('======================================')            
            
                fitness_value_0= mutation_cost(mutated_child_0) 
                fitness_value_1= mutation_cost(mutated_child_1)
                
                if fitness_value_0 < fitness_value_1:
                    test_dict[g][i] = mutated_child_0
                else:
                    test_dict[g][i] = mutated_child_1
            
          #  min_fit = min(fitness_value_0, fitness_value_1 )
            
            #list_fitness.append(min_fit)
            #print('G:', g, ' N:', i, 'L3', min_fitness)
            
                min_fitness[g][i] = min(fitness_value_0, fitness_value_1)


# start from the first row and keep the min value for each g 
    min_cost = min_fitness[1][min(min_fitness[1].keys(), key=(lambda k: min_fitness[1][k]))]
    min_population= (test_dict[1][min(min_fitness[1].keys(), key=(lambda k: min_fitness[1][k]))])
    
    for g in range (1, g_max+1):
        # we keep this line out of if statement to capture the minimum for each g , and if it goes into if statement it compares each time 
        min_list_cost.append(min_fitness[g][min(min_fitness[g].keys(), key=(lambda k: min_fitness[g][k]))])
        if min_fitness[g][min(min_fitness[g].keys(), key=(lambda k: min_fitness[g][k]))] <= min_cost:
            min_cost = min_fitness[g][min(min_fitness[g].keys(), key=(lambda k: min_fitness[g][k]))]
            min_population = (test_dict[g][min(min_fitness[g].keys(), key=(lambda k: min_fitness[g][k]))])
#
    return min_cost, min_population, min_list_cost             

 #%% visualization of the result
int_solution = initialize (['D2','D10','D4','D1','D11', 'D12','D3', 'D9', 'D6','D7', 'D5','D8'])
a,b,c = ga_qap(50, 180, 0.8, 0.9, int_solution)

df = pd.DataFrame({'fitness_value':c})

df['generation'] = range(1,len(df)+1)

df.plot(kind='line',x='generation',y='fitness_value', color = 'r', title = 'REPL-3: Population Size = 120')
print(a,b)  
    
