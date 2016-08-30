"""
  MIP for Boolean feature selection.

  Author: Shannon Fenn (shannon.fenn@gmail.com)
"""
import cplex
import numpy as np


def single_minimum_feature_set(features, target):
    if np.all(target) or not np.any(target):
        # constant target - no solutions
        return []
    coverage = build_coverage(features, target)
    _, solution = mink(coverage)
    return solution


# def all_minimum_feature_sets(features, target):
#     if np.all(target) or not np.any(target):
#         # constant target - no solutions
#         return []
#     coverage = build_coverage(features, target)
#     return all_minfs(coverage)


def build_coverage(features, target):
    Ne, Nf = features.shape
    class_0_indices = np.flatnonzero(target == 0)
    class_1_indices = np.flatnonzero(target)
    num_eg_pairs = class_0_indices.size * class_1_indices.size
    coverage_matrix = np.zeros((num_eg_pairs, Nf), dtype=float)
    i = 0
    for i0 in class_0_indices:
        for i1 in class_1_indices:
            for f in range(Nf):
                if features[i0, f] != features[i1, f]:
                    coverage_matrix[i, f] = 1
            i += 1
    return coverage_matrix


def mink(coverage):
    Np, Nf = coverage.shape

    model = cplex.Cplex()

    # stop cplex chatter
    model.set_results_stream(None)
    model.set_warning_stream(None)
    model.set_error_stream(None)
    model.set_log_stream(None)

    model.objective.set_sense(model.objective.sense.minimize)

    obj_coefficients = [1] * Nf
    lbounds = [0] * Nf
    ubounds = [1] * Nf
    types = 'I' * Nf
    feature_ids = list(range(Nf))

    sense = 'G' * Np
    rhs = [1] * Np

    model.variables.add(
        obj=obj_coefficients, lb=lbounds, ub=ubounds, types=types)#, names=my_colnames)

    rows = [[feature_ids, row] for row in coverage]

    model.linear_constraints.add(
        lin_expr=rows, senses=sense, rhs=rhs)#, names=my_rownames)

    model.solve()

    if model.solution.get_status() == model.solution.status.MIP_optimal:
        x = model.solution.get_values()
        fs = np.where(x)[0].tolist()
        return len(fs), fs
    else:
        return 0, []


# def all_kfs(coverage, k):
#     Np, Nf = coverage.shape
#     solver = pywrapcp.Solver("all_kfs")

#     # decision variable
#     x = [solver.IntVar(0, 1, 'x[{}]'.format(f)) for f in range(Nf)]

#     # constraints
#     # all pairs must be covered by at least one feature
#     for p in range(Np):
#         b = [x[f] for f in range(Nf) if coverage[p][f]]
#         solver.Add(solver.SumGreaterOrEqual(b, 1))
#     # only k features may be selected
#     solver.Add(solver.SumEquality(x, k))

#     # solution and search
#     solution = solver.Assignment()
#     solution.Add(x)
#     # solution.AddObjective(k)

#     collector = solver.AllSolutionCollector(solution)
#     db = solver.Phase(x,
#                       solver.INT_VAR_DEFAULT,
#                       solver.INT_VALUE_DEFAULT)
#     solution_found = solver.Solve(db, collector)

#     if solution_found:
#         # collect all feature sets
#         numSol = collector.SolutionCount()
#         feature_sets = [[f for f in range(Nf) if collector.Value(i, x[f]) == 1]
#                         for i in range(numSol)]

#         return feature_sets
#     else:
#         return []


# def all_minfs(coverage):
#     Np, Nf = coverage.shape
#     solver = pywrapcp.Solver("all_minfs")

#     # decision variable
#     x = [solver.IntVar(0, 1, 'x[{}]'.format(f)) for f in range(Nf)]

#     # constraints
#     # all pairs must be covered by at least one feature
#     for p in range(Np):
#         b = [x[f] for f in range(Nf) if coverage[p][f]]
#         solver.Add(solver.SumGreaterOrEqual(b, 1))

#     # objective
#     # minimise number of features
#     k = solver.Sum(x)
#     objective = solver.Minimize(k, 1)

#     # find min k
#     solution = solver.Assignment()
#     solution.Add(x)
#     solution.AddObjective(k)

#     collector = solver.LastSolutionCollector(solution)
#     db = solver.Phase(x + [k], solver.INT_VAR_DEFAULT, solver.INT_VALUE_DEFAULT)
#     solution_found = solver.Solve(db, [collector, objective])

#     if not solution_found:
#         return []

#     # find all kfs
#     # only k features may be selected
#     min_k = collector.ObjectiveValue(0)
#     solver.Add(solver.SumEquality(x, min_k))

#     # solution and search
#     collector = solver.AllSolutionCollector(solution)
#     db = solver.Phase(x, solver.INT_VAR_DEFAULT, solver.INT_VALUE_DEFAULT)
#     solution_found = solver.Solve(db, collector)

#     if not solution_found:
#         return []

#     # collect all feature sets
#     numSol = collector.SolutionCount()
#     feature_sets = [[f for f in range(Nf) if collector.Value(i, x[f]) == 1]
#                     for i in range(numSol)]

#     return feature_sets
