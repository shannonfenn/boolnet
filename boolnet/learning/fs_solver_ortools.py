"""
  Set covering for Boolean feature selection.

  Author: Shannon Fenn (shannon.fenn@gmail.com)
"""
from ortools.constraint_solver import pywrapcp
import numpy as np


def single_minimum_feature_set(features, target):
    if np.all(target) or not np.any(target):
        # constant target
        return [list(range(features.shape[1]))]
    coverage = build_coverage(features, target)
    _, solution = mink(coverage)
    return solution


def all_minimum_feature_sets(features, target):
    if np.all(target) or not np.any(target):
        # constant target
        return [list(range(features.shape[1]))]
    coverage = build_coverage(features, target)
    return all_minfs(coverage)


def build_coverage(features, target):
    Ne, Nf = features.shape
    class_0_indices = np.flatnonzero(target == 0)
    class_1_indices = np.flatnonzero(target)
    num_eg_pairs = class_0_indices.size * class_1_indices.size
    coverage_matrix = np.zeros((num_eg_pairs, Nf), dtype=np.uint8)
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
    solver = pywrapcp.Solver("mink")

    # decision variable
    x = [solver.IntVar(0, 1, 'x[{}]'.format(f)) for f in range(Nf)]

    # constraints
    # all pairs must be covered by at least one feature
    for p in range(Np):
        b = [x[f] for f in range(Nf) if coverage[p][f]]
        solver.Add(solver.SumGreaterOrEqual(b, 1))

    # objective
    # minimise number of features
    k = solver.Sum(x)
    objective = solver.Minimize(k, 1)

    # solution and search
    solution = solver.Assignment()
    solution.Add(x)
    solution.AddObjective(k)

    collector = solver.LastSolutionCollector(solution)
    solution_found = solver.Solve(solver.Phase(x + [k],
                                               solver.INT_VAR_DEFAULT,
                                               solver.INT_VALUE_DEFAULT),
                                  [collector, objective])

    if solution_found:
        best_k = collector.ObjectiveValue(0)
        best_indices = [f for f in range(Nf) if collector.Value(0, x[f]) == 1]
        return best_k, best_indices
    else:
        return 0, []


def all_kfs(coverage, k):
    Np, Nf = coverage.shape
    solver = pywrapcp.Solver("all_kfs")

    # decision variable
    x = [solver.IntVar(0, 1, 'x[{}]'.format(f)) for f in range(Nf)]

    # constraints
    # all pairs must be covered by at least one feature
    for p in range(Np):
        b = [x[f] for f in range(Nf) if coverage[p][f]]
        solver.Add(solver.SumGreaterOrEqual(b, 1))
    # only k features may be selected
    solver.Add(solver.SumEquality(x, k))

    # solution and search
    solution = solver.Assignment()
    solution.Add(x)
    # solution.AddObjective(k)

    collector = solver.AllSolutionCollector(solution)
    db = solver.Phase(x,
                      solver.INT_VAR_DEFAULT,
                      solver.INT_VALUE_DEFAULT)
    solution_found = solver.Solve(db, collector)

    if solution_found:
        # collect all feature sets
        numSol = collector.SolutionCount()
        feature_sets = [[f for f in range(Nf) if collector.Value(i, x[f]) == 1]
                        for i in range(numSol)]

        return feature_sets
    else:
        return []


def all_minfs(coverage):
    Np, Nf = coverage.shape
    solver = pywrapcp.Solver("all_minfs")

    # decision variable
    x = [solver.IntVar(0, 1, 'x[{}]'.format(f)) for f in range(Nf)]

    # constraints
    # all pairs must be covered by at least one feature
    for p in range(Np):
        b = [x[f] for f in range(Nf) if coverage[p][f]]
        solver.Add(solver.SumGreaterOrEqual(b, 1))

    # objective
    # minimise number of features
    k = solver.Sum(x)
    objective = solver.Minimize(k, 1)

    # find min k
    solution = solver.Assignment()
    solution.Add(x)
    solution.AddObjective(k)

    collector = solver.LastSolutionCollector(solution)
    db = solver.Phase(x + [k], solver.INT_VAR_DEFAULT, solver.INT_VALUE_DEFAULT)
    solution_found = solver.Solve(db, [collector, objective])

    if not solution_found:
        return []

    # find all kfs
    # only k features may be selected
    min_k = collector.ObjectiveValue(0)
    solver.Add(solver.SumEquality(x, min_k))

    # solution and search
    collector = solver.AllSolutionCollector(solution)
    db = solver.Phase(x, solver.INT_VAR_DEFAULT, solver.INT_VALUE_DEFAULT)
    solution_found = solver.Solve(db, collector)

    if not solution_found:
        return []

    # collect all feature sets
    numSol = collector.SolutionCount()
    feature_sets = [[f for f in range(Nf) if collector.Value(i, x[f]) == 1]
                    for i in range(numSol)]

    return feature_sets
