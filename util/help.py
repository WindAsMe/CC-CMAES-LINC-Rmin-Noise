import numpy as np
import random


def random_Population(scale_range, N, size):
    Population = np.zeros((size, N), dtype='double')
    for individual in Population:
        for i in range(len(individual)):
            individual[i] = random.uniform(scale_range[0], scale_range[1])
    return Population


def empty_groups(size):
    groups = []
    for i in range(size):
        groups.append([])
    return groups


def Phen_Groups(Phen, Empty_groups):
    for i in range(len(Phen)):
        Empty_groups[Phen[i]-1].append(i)
    # Remove the empty groups
    empty_index = []
    for j in range(len(Empty_groups)):
        if not Empty_groups[j]:
            empty_index.append(j)
    if len(empty_index) != 0:
        Empty_groups = list(np.delete(Empty_groups, empty_index, axis=0))
    return Empty_groups
