"""
  Set covering for Boolean feature selection.

  Author: Shannon Fenn (shannon.fenn@gmail.com)
"""
import Numberjack as nj
import numpy as np


def all_minimum_feature_sets(features, target):
    if np.all(target) or not np.any(target):
        # constant target
        return [list(range(features.shape[1]))]
    coverage = build_coverage(features, target)
    k, _ = mink(coverage)
    if k > 0:
        return all_kfs(coverage, k)
    else:
        return []


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

    # binary decision variables
    x = [nj.Variable() for f in range(Nf)]

    model = nj.Model()
    for p in range(Np):
        model.add(nj.Sum(x[f] for f in range(Nf) if coverage[p][f]) > 0)

    # objective
    # minimise number of features
    k = nj.Sum(x)
    model.add(nj.Minimise(k))

    # solution
    solver = model.load('Mistral')
    satisfiable = solver.solve()

    # return results
    if satisfiable:
        best_k = k.get_value()
        best_indices = [f for f in range(Nf) if x[f].get_value()]
        return best_k, best_indices
    else:
        return 0, []


def all_kfs(coverage, k):
    Np, Nf = coverage.shape

    # binary decision variables
    x = [nj.Variable() for f in range(Nf)]

    # constraints
    # all pairs must be covered by at least one feature
    model = nj.Model()
    for p in range(Np):
        model.add(nj.Sum(x[f] for f in range(Nf) if coverage[p][f]) > 0)
    # only k features may be selected
    model.add(nj.Sum(x) == k)

    # solution
    solver = model.load('Mistral')
    satisfiable = solver.solve()

    if satisfiable:
        feature_sets = [[f for f in range(Nf) if x[f].get_value()]]
        while solver.getNextSolution():
            feature_sets.append([f for f in range(Nf) if x[f].get_value()])
        return feature_sets
    else:
        return []
