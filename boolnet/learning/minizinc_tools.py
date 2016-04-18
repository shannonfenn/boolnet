import subprocess as sp


def mzn_all_minfs(coverage_matrix):
    k = mzn_min_k(coverage_matrix)
    return mzn_all_kfs(coverage_matrix, k)


def mzn_min_k(coverage_matrix):
    model_name = 'mink.mzn'

    data_string = 'Npairs = {}; Nf = {};'.format(*coverage_matrix.shape)
    data_string += 'coverage = [|' + '|'.join(
        ','.join(str(v) for v in row) for row in coverage_matrix) + '|];'

    cmd_string = ['mzn-gecode', model_name, '-D', data_string,
                  '--soln-sep=\'\'', '--search-complete-msg=\'\'']

    # run minizinc model
    output = sp.run(cmd_string, stdout=sp.PIPE, check=True,
                    universal_newlines=True)

    k = int(output.stdout.split()[0])

    return k


def mzn_all_kfs(coverage_matrix, k):
    model_name = 'kfs.mzn'

    Npairs, Nf = coverage_matrix.shape

    data_string = 'Npairs = {}; Nf = {}; k = {};'.format(Npairs, Nf, k)
    data_string += 'coverage = [|' + '|'.join(
        ','.join(str(v) for v in row) for row in coverage_matrix) + '|];'

    cmd_string = ['mzn-gecode', model_name, '-D', data_string, '-a',
                  '--soln-sep=', '--search-complete-msg=']

    # run minizinc model
    output = sp.run(cmd_string, stdout=sp.PIPE, check=True,
                    universal_newlines=True)

    feature_sets = [[int(c) for c in line.strip('[] ').split(',')]
                    for line in output.stdout.splitlines() if line]

    return feature_sets
