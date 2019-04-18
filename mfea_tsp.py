import bisect
from math import * 
import numpy as np
import random
import matplotlib.pyplot as plt
class City:
    def __init__(self, x, y):

        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Task(object):
    """docstring for Task"""
    def __init__(self, cityList):
        super(Task, self).__init__()
        self.cityList = cityList
        self.len = len(cityList)

tasks = []
cityList = []
for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
tasks.append(Task(cityList))
cityList = []
for i in range(0,20):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
tasks.append(Task(cityList))
cityList = []
for i in range(0,18):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
tasks.append(Task(cityList))
class Individual(object):
    """docstring for """
    def __init__(self, route):
        super(Individual, self).__init__()
        self.route = route
        self.skill_factor = 0
        self.scalar_fitness = 1e9
        self.fitness = [1e9]*len(tasks)
        self.rank = [0]*len(tasks)
    def routeDistance(self,task_id):
        task = tasks[task_id]
        route = [i for i in self.route if i in list(range(len(task.cityList)))]
        pathDistance = 0
        for i in range(0, len(task.cityList)):
            fromCity = task.cityList[route[i]]
            toCity = None
            if i + 1 < len(task.cityList):
                toCity = task.cityList[route[i + 1]]
            else:
                toCity = task.cityList[0]
            pathDistance += fromCity.distance(toCity)
        self.fitness[task_id] = pathDistance

    def evaluate_all(self):
        for i in range(len(tasks)):
            self.routeDistance(i)
    def evaluate(self,task_no):
            self.routeDistance(task_no)
    def __str__(self):
        return str(self.route)

def mutate(parent, mutationRate=0.2):
    individual = parent.route
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    child = Individual(individual)
    child.skill_factor = parent.skill_factor
    child.evaluate(parent.skill_factor)
    return child

def crossover(Iparent1, Iparent2):
    parent1 = Iparent1.route
    parent2 = Iparent2.route
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    Child = Individual(child)
    if (np.random.rand()<0.5):
        Child.skill_factor = Iparent1.skill_factor
    else:
        Child.skill_factor = Iparent2.skill_factor
    Child.evaluate(Child.skill_factor)
    return Child
popsize = 100
num_of_generation = 1000
rcm = 0.3
mutation_rate=0.1
citySize = []
progress = [[],[],[]]
for i in tasks:
    citySize.append(i.len)
geneSize = max(citySize)
population = []
for i in range(0, popsize):
    indiv = Individual(random.sample(list(range(geneSize)), geneSize))
    indiv.evaluate_all()
    population.append(indiv)
#tao dan so
for i in range(len(tasks)):
    population.sort(key=lambda x: x.fitness[i])
    for j,indiv in enumerate(population):
        indiv.rank[i]=j
    print(population[0].fitness[i])
    progress[i].append(population[0].fitness[i])
# sap xep rank 
for j,indiv in enumerate(population):
    indiv.scalar_fitness = min(indiv.rank)
    indiv.skill_factor = indiv.rank.index(indiv.scalar_fitness)
# caculate skill factor
map=[0,0,0]
for gen in range(num_of_generation):
    parent1, parent2 =  random.sample(population,2)
    child = []
    top = [-1e9]*len(tasks)
    for _ in range(int(popsize/2)):
        if (np.random.rand() < rcm) or (parent1.skill_factor == parent2.skill_factor):
            child.append(crossover(parent1,parent2))
            child.append(crossover(parent1,parent2))
        else:
            child.append(mutate(parent1,0.3))
            child.append(mutate(parent2,0.3))
        temp = population + child
    for i in range(len(tasks)):
        temp.sort(key=lambda x: x.fitness[i])
        top[i]=str(temp[0])+str(temp[0].fitness[i])
        for j,indiv in enumerate(temp):
            indiv.rank[i]=j
        print(temp[0].fitness[i])
        progress[i].append(temp[0].fitness[i])
        map[i]=temp[0]

    # recaculate rank
    for j,indiv in enumerate(temp):
        indiv.scalar_fitness = min(indiv.rank)
        indiv.skill_factor = indiv.rank.index(indiv.scalar_fitness)
    # recaculate skill factor
    temp.sort(key=lambda x: x.scalar_fitness)
    population=temp[:popsize]

    # if gen%100 == 1:
    #     print ("current besst task1 :",map[0].fitness[0])
    #     x = [tasks[0].cityList[i].x for i in map[0].route]
    #     y = [tasks[0].cityList[i].y for i in map[0].route]
    #     plt.plot(x,y,zorder=2)
    #     plt.scatter(x,y,zorder=1)
    #     plt.title('task1')
    #     plt.show()

plt.plot(progress[0])
plt.plot(progress[1])
plt.plot(progress[2])
plt.ylabel('Distance')
plt.xlabel('Generation')
plt.show()

x = [tasks[0].cityList[i].x for i in map[0].route]
y = [tasks[0].cityList[i].y for i in map[0].route]
plt.plot(x,y,zorder=2)
plt.scatter(x,y,zorder=1)
plt.title('task1')
plt.show()
map1 = [i for i in map[1].route if i in list(range(20))]
print (map1)
x = [tasks[1].cityList[i].x for i in map1]
y = [tasks[1].cityList[i].y for i in map1]
plt.plot(x,y,zorder=2)
plt.scatter(x,y,zorder=1)
plt.title('task2')
plt.show()

map1 = [i for i in map[2].route if i in list(range(18))]
print (map1)
x = [tasks[2].cityList[i].x for i in map1]
y = [tasks[2].cityList[i].y for i in map1]
plt.plot(x,y,zorder=2)
plt.scatter(x,y,zorder=1)
plt.title('task3')

plt.show()
# def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
#     pop = initialPopulation(popSize, population)
#     progress = []
#     progress.append(1 / rankRoutes(pop)[0][1])
    
#     for i in range(0, generations):
#         pop = nextGeneration(pop, eliteSize, mutationRate)
#         progress.append(1 / rankRoutes(pop)[0][1])
    
#     plt.plot(progress)
#     plt.ylabel('Distance')
#     plt.xlabel('Generation')
#     plt.show()
# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
