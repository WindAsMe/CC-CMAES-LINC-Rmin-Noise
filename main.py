import numpy as np
from cec2013lsgo.cec2013 import Benchmark
from CMAES.optimizer import CC_CMAES
from Grouping.Comparison import DECC_DG, DECC_D, DECC_G, CCVIl
from os import path


def write_obj(data, path):
    with open(path, 'a+') as f:
        f.write(str(data) + ', ')
        f.write('\n')
        f.close()


def valid_groups(groups):
    sub_groups = []
    combined_group = []
    for group in groups:
        if len(group) == 1:
            combined_group.extend(group)
        else:
            sub_groups.append(group)
    sub_groups.append(combined_group)
    return sub_groups


if __name__ == "__main__":
    Gene_len = 8
    Dim = 1000
    this_path = path.realpath(__file__)

    bench = Benchmark()
    func_num = 1
    NIND = 30
    FEs = 3000000
    func = bench.get_function(func_num)
    info = bench.get_info(func_num)
    var_range = [info["lower"], info["upper"]]
    trial_run = 1

    CCVIL_obj_path = path.dirname(this_path) + "/Data/obj/CCVIL/f" + str(func_num)
    D_obj_path = path.dirname(this_path) + "/Data/obj/D/f" + str(func_num)
    DG_obj_path = path.dirname(this_path) + "/Data/obj/DG/f" + str(func_num)
    G_obj_path = path.dirname(this_path) + "/Data/obj/G/f" + str(func_num)
    GA_obj_path = path.dirname(this_path) + "/Data/obj/proposal/f" + str(func_num)

    for i in range(trial_run):
        print("trial run: ", i)
        CCVIL_groups, CCVIL_cost = CCVIl(Dim, func)
        D_groups = DECC_D(Dim, func, var_range, groups_num=20, max_number=50)
        DG_groups, DG_cost = DECC_DG(Dim, func)
        G_groups = DECC_G(Dim, groups_num=20, max_number=50)

        CCVIL_Max_iter = int((FEs - CCVIL_cost) / NIND / Dim) - 10
        CCVIL_obj_trace = CC_CMAES(Dim, CCVIL_groups, NIND, CCVIL_Max_iter, func, var_range)
        write_obj(CCVIL_obj_trace, CCVIL_obj_path)

        D_Max_iter = int((FEs - 30000) / NIND / Dim) - 10
        D_obj_trace = CC_CMAES(Dim, D_groups, NIND, D_Max_iter, func, var_range)
        write_obj(D_obj_trace, D_obj_path)

        DG_Max_iter = int((FEs - DG_cost) / NIND / Dim) - 10
        DG_obj_trace = CC_CMAES(Dim, DG_groups, NIND, DG_Max_iter, func, var_range)
        write_obj(DG_obj_trace, DG_obj_path)

        G_Max_iter = int(FEs / NIND / Dim) - 10
        G_obj_trace = CC_CMAES(Dim, G_groups, NIND, G_Max_iter, func, var_range)
        write_obj(G_obj_trace, G_obj_path)

