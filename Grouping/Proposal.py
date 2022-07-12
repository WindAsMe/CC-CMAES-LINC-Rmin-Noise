from util import help
from Grouping import optimizer, Comparison


def MALINC_Rmin(Dim, Gene_len, func, pop_size, scale_range, cost, intercept):
    """
    Algorithm initialization
    """
    NIND = 20
    Max_iter = 20
    random_Pop = help.random_Population(scale_range, Dim, pop_size)

    """
    Apply GA
    """
    Groups, obj, temp_cost = optimizer.MA_optmize(Dim, Gene_len, func, random_Pop, NIND, Max_iter, intercept)
    cost += temp_cost
    return Groups, cost
