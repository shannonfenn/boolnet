"""
  Set covering for Boolean feature selection.

  Author: Shannon Fenn (shannon.fenn@gmail.com)
"""
from ortools.constraint_solver import pywrapcp
import numpy as np


def all_minimum_feature_sets(features, target):
    coverage = build_coverage(features, target)
    k, _ = mink(coverage)
    return all_kfs(coverage, k)


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
    solver.Solve(solver.Phase(x + [k],
                              solver.INT_VAR_DEFAULT,
                              solver.INT_VALUE_DEFAULT),
                 [collector, objective])

    best_k = collector.ObjectiveValue(0)
#    best_x = [collector.Value(0, x[f]) for f in range(Nf)]
    best_indices = [f for f in range(Nf) if collector.Value(0, x[f]) == 1]
    return best_k, best_indices


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
    solver.Solve(db, collector)

    # collect all feature sets
    numSol = collector.SolutionCount()
#    feature_sets = [[collector.Value(i, v) for v in x] for i in range(numSol)]
    feature_sets = [[f for f in range(Nf) if collector.Value(i, x[f]) == 1]
                    for i in range(numSol)]

    return feature_sets
