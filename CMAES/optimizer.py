import numpy as np
from cmaes import CMA
from sklearn.preprocessing import MinMaxScaler
import geatpy as ea


def CC_CMAES(Dim, groups, NIND, Max_iter, func, var_range):
    var_traces = np.zeros((Max_iter, Dim))
    initial_pops, base_pop = pop_initial(Dim, groups, var_range)
    real_iter = 0

    while real_iter < Max_iter:
        for i in range(len(groups)):
            initial_pops[i] = sub_CMAES(NIND, base_pop, groups[i], initial_pops[i], func, var_range)

            for j in range(len(groups[i])):
                var_traces[real_iter][groups[i][j]] = initial_pops[i][j]
        real_iter += 1

    obj_traces = []
    for var in var_traces:
        obj_traces.append(func(var) * (1 + np.random.normal(loc=0, scale=0.01, size=None)))
    for i in range(1, len(obj_traces)):
        if obj_traces[i] > obj_traces[i-1]:
            obj_traces[i] = obj_traces[i-1]
    return obj_traces


def sub_CMAES(NIND, base_pop, group, pop, func, var_range):
    scale_range = np.array([var_range] * len(group))
    cma_es = CMA(mean=pop, sigma=1.3, bounds=scale_range, population_size=NIND*len(group))

    best_indi = None

    for generation in range(1):
        solutions = []
        for _ in range(cma_es.population_size):
            x = cma_es.ask()
            s = extend(x, base_pop, group)
            value = func(s) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
            solutions.append((x, value))
        cma_es.tell(solutions)
        elites_len = max(int(len(solutions) * 0.01), 1)
        elites = np.array(sort(solutions)[0:elites_len], dtype='object')
        best_indi = DynamicAverage(group, base_pop, elites, func, NIND * len(group) / 10)
    return best_indi


def sort(solutions):
    for i in range(len(solutions)):
        for j in range(i+1, len(solutions)):
            if solutions[i][1] > solutions[j][1]:
                temp = solutions[i]
                solutions[i] = solutions[j]
                solutions[j] = temp
    return solutions


def DynamicAverage(group, base_pop, elites, func, computation):

    scaler = MinMaxScaler()
    FitV = ea.scaling(np.array(elites[:, 1]).reshape(-1, 1))

    trans_FitV = scaler.fit_transform(FitV)[:, 0]
    amp_FitV = amplify(trans_FitV)
    reevaluation = resourceAssign(amp_FitV, computation)

    ObjV = []
    for i in range(len(elites)):
        obj = 0
        for time in range(int(reevaluation[i])):
            obj += func(extend(elites[i][0], base_pop, group)) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
        ObjV.append((obj + elites[i][1]) / (reevaluation[i] + 1))
    best_indi = elites[np.argmin(ObjV)][0]
    return best_indi


def amplify(FitV):
    amp_FitV = []
    nomi = 0
    for fit in FitV:
        nomi += fit
    for fit in FitV:
        amp_FitV.append(fit / nomi)
    return amp_FitV


# Roulette allocate
def resourceAssign(amp_FitV, computational):
    allocation = [0] * len(amp_FitV)
    accumulative_P = [0]
    for i in range(1, len(amp_FitV)):
        accumulative_P.append(accumulative_P[i-1] + amp_FitV[i-1])
    i = 0
    while i < computational:
        r = np.random.rand()
        ind = locate(accumulative_P, r)
        allocation[ind] += 1
        i += 1
    return allocation


def locate(P, r):
    for i in range(len(P) - 1):
        if P[i] <= r <= P[i+1]:
            return i
    return len(P) - 1


def elite_extract(elite_index, Phen, ObjV, FitV):
    elite_Phen = []
    elite_FitV = []
    elite_ObjV = []
    for i in elite_index:
        elite_Phen.append(Phen[i])
        elite_FitV.append([FitV[i]])
        elite_ObjV.append(ObjV[i])
    return elite_Phen, elite_ObjV, elite_FitV


def extend(x, base_pop, group):
    for i in range(len(group)):
        base_pop[group[i]] = x[i]
    return base_pop


def pop_initial(Dim, groups, var_range):
    pops = []
    base_pop = np.zeros(Dim)
    for group in groups:
        pop = (var_range[1] - var_range[0]) * np.random.rand(len(group)) + var_range[0]
        pops.append(pop)
        for i in range(len(group)):
            base_pop[group[i]] = pop[i]
    return pops, base_pop


