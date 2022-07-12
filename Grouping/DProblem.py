import geatpy as ea
from util import help
import numpy as np
from bench import benchmark


class MyProblem(ea.Problem):
    def __init__(self, Dim, gene_len, func, random_Pop, intercept, base_fitness):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.random_Pop = random_Pop
        self.intercept = intercept
        self.cost = 0
        self.Gene_len = gene_len
        self.base_fitness = base_fitness
        self.func = func
        self.Dim = Dim
        self.varTypes = [1] * Dim
        self.lb = [0] * Dim
        self.ub = [2 ** gene_len-1] * Dim
        self.lbin = [1] * Dim
        self.ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, self.varTypes, self.lb, self.ub, self.lbin, self.ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen
        Objs = []
        for var in Vars:
            # Decompose the problem depending on
            Empty_groups = help.Phen_Groups(var, help.empty_groups(2 ** self.Gene_len))

            groups_fitness, self.cost = benchmark.groups_fitness(Empty_groups, self.random_Pop, self.func, self.cost,
                                                                 self.intercept)
            Objs.append([benchmark.object_function(self.base_fitness, groups_fitness, len(Empty_groups))])
        pop.ObjV = np.array(Objs, dtype='double')





