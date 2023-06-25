# quadratic_assignment_problem_GA
Solving quadratic assignment problems using genetic algorithm

QAP with GA: 

Step 1) Permutation Encoding and Generate Chromosome: First, we need to encode the 
individuals or solutions in the population in the genetic algorithm. In the permutation encoding, 
each chromosome will be shown as a string of department assignments and each gene represents 
a position in this string. So, we create a function that creates a chromosome with a length of 12 to 
generate the initial population. This means each chromosome has 12 genes and each gene (D1, D2, 
…) represents the assignment of each department to exactly one location. In Table 2, for example, 
chromosome D[11,6,9,4,12,2,8, 3,7,10,1,5] means that departments 11, 6, 9, .. have been assigned to 
locations 1, 2, 3, … respectively. 
9 
Step 2) Crossover: In this step, I used a single-point crossover to make children. In fact, one point 
is randomly selected, and the tails of each parent are swapped with each other with a probability of 
crossover (Pc). As we prefer to have a high probability of crossover I have changed this probability 
by values 0.2, 0.5, 0.77, and 1.0 as shown in Table 2. 

Step 3) Mutation - Inversion; The mutation operator changes the value of each gene randomly 
by inverting the values in a chunk of the chromosome. First, we generate two random points and 
inverse the order of values in the middle section. 

Step 4) Tournament Selection (k = 3); In the deterministic tournament selection, the candidate 
the solution will be picked randomly out of the generated population and the best candidate which has 
the best fitness value will be selected and stored as the first parent. In the second run, again the 
tournament size of 3 (k=3) will be picked randomly and the best one will become the second 
parent. 

Step 5) The stopping criteria: the algorithm running ends after it goes over the determined number 
of generations which is equal to 250. In the end, it declares the minimum fitness value for the flow 
cost function. 
