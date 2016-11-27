import localsolver as ls


def is_feature_set(M, T):
    class_0_indices = np.flatnonzero(T == 0)
    class_1_indices = np.flatnonzero(T)
    for i0 in class_0_indices:
        for i1 in class_1_indices:
            if np.array_equal(M[i0], M[i1]):
                return False
    return True


def build_model(M, T, model):
    Ne, Nf = M.shape
    x = [model.bool() for f in range(Nf)]

    for p in range(Ne):
        for q in range(Ne):
            if p != q and T[p] != T[q]:
                sum_xa = model.sum(x[f] for f in range(Nf) if M[p][f] != M[q][f])
                model.constraint(sum_xa >= 1)

    k = model.sum(x)
    model.minimize(k)
    model.close()
    return x


def single_minimum_feature_set(M, T, prior_soln=None, timelimit=None, debug=False):
    if not (0 < T.sum() < len(T)):
        # constant target
        return []

    with ls.LocalSolver() as solver:
        model = solver.model

        X = build_model(M, T, model)

        if not debug:
            solver.param.verbosity = 0

        if timelimit > 0:
            solver.create_phase().time_limit = timelimit

        if prior_soln is not None:
            for x in X:
                x.value = 0
            for i in prior_soln:
                X[i].value = 1

        solver.solve()

        solution = solver.solution

        status = solution.status

        if status == ls.LSSolutionStatus.INCONSISTENT:
            raise ValueError('Invalid instance:\n{}\n{}'.format(M, T))
        elif status == ls.LSSolutionStatus.INFEASIBLE:
            # test if full solution is feasible
            if is_feature_set(M, T):
                return list(range(len(T)))
            else:
                raise ValueError('Invalid instance:\n{}\n{}'.format(M, T))
        elif status == ls.LSSolutionStatus.FEASIBLE:
            # May do something different later
            return [i for i, x in enumerate(X) if x.value == 1]
        elif status == ls.LSSolutionStatus.OPTIMAL:
            return [i for i, x in enumerate(X) if x.value == 1]
