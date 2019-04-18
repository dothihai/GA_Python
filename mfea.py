import bisect
from math import * 
import numpy as np
import random
#bisect.insort(list, n)   chen vao sorted list	
def task1(gene):
	x = 0
	for bit in gene: x=x*2+bit
	x = -1+x*(2./2**len(gene))
	return x*sin(pi*x*10)+1
def task2(gene):
	x = 0
	for bit in gene: x=x*2+bit
	x = -1+x*(2./2**len(gene))
	return x*cos(pi*x*10)
def task3(gene):
	x = 0
	for bit in gene: x=x*2+bit
	x = -1+x*(2./2**len(gene))
	return x*sin(pi*x*4+1)
def task4(gene):
	x = 0
	for bit in gene: x=x*2+bit
	x = -1+x*(2./2**len(gene))
	return x*sin(pi*x*4+5)
tasks = [task1,task2,task3,task4]
num_of_task = len(tasks)
popsize = 30
num_of_gene = 22
num_of_generation = 1000
rcm = 0.2
mutation_rate=0.1
# constant 
class Invidual(object):
	"""docstring for Invidual"""
	def __init__(self, gene):
		super(Invidual, self).__init__()
		self.gene = gene
		self.skill_factor = 0
		self.scalar_fitness = 1e9
		self.fitness = [-1e9]*num_of_task
		self.rank = [0]*num_of_task
	def evaluate_all(self):
		for i in range(len(tasks)):
			self.fitness[i]=tasks[i](self.gene)
	def evaluate(self,task_no):
		self.fitness[task_no] = tasks[task_no](self.gene)
	def __str__(self):
		return str(self.gene)

def crossover(x,y):
	cut = int(np.random.randint(num_of_gene))
	child = Invidual(np.concatenate((x.gene[cut:],y.gene[:cut])))
	if (np.random.rand()<0.5):
		child.skill_factor = x.skill_factor
	else:
		child.skill_factor = y.skill_factor
	child.evaluate(child.skill_factor)
	return child

def mutate(x):
	child = Invidual(x.gene)
	for i in range(num_of_gene):
		if (np.random.rand()<mutation_rate):
			child.gene[i] = child.gene[i] ^ 1
	child.skill_factor = x.skill_factor
	child.evaluate(x.skill_factor)
	return child

current_population = []
for _ in range(popsize):
	new = Invidual(np.random.randint(0,2,num_of_gene))
	current_population.append(new)
	new.evaluate_all()
# generate populaton
for i in range(len(tasks)):
	current_population.sort(key=lambda x: -x.fitness[i])
	# print (current_population[0].fitness[i],current_population[0].gene)
	for j,indiv in enumerate(current_population):
		indiv.rank[i]=j
# caculate rank
for j,indiv in enumerate(current_population):
	indiv.scalar_fitness = min(indiv.rank)
	indiv.skill_factor = indiv.rank.index(indiv.scalar_fitness)
# caculate skill factor
for gen in range(num_of_generation):
	parent1, parent2 =  random.sample(current_population,2)
	child = []
	top = [-1e9]*num_of_task
	for _ in range(int(popsize/2)):
		if (np.random.rand() < rcm) or (parent1.skill_factor == parent2.skill_factor):
			child.append(crossover(parent1,parent2))
			child.append(crossover(parent1,parent2))
		else:
			child.append(mutate(parent1))
			child.append(mutate(parent2))
		temp = current_population + child
	for i in range(len(tasks)):
		temp.sort(key=lambda x: -x.fitness[i])
		top[i]=str(temp[0])+str(temp[0].fitness[i])
		for j,indiv in enumerate(temp):
			indiv.rank[i]=j
	# recaculate rank
	for j,indiv in enumerate(temp):
		indiv.scalar_fitness = min(indiv.rank)
		indiv.skill_factor = indiv.rank.index(indiv.scalar_fitness)
	# recaculate skill factor
	temp.sort(key=lambda x: x.scalar_fitness)
	current_population=temp[:popsize]
	# for x in temp:
	# 	print(x,end = ' ')
	# print()
	if gen%20==0: 
		print(gen)
		for i in top:
			print(i)
