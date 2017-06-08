import argparse
import glob
from os.path import isdir, expanduser, abspath, splitext, join
from natsort import natsorted


def directory_type(directory):
    # Handle tilde
    directory = abspath(expanduser(directory))
    if isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid directory.'.format(directory))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir', type=directory_type)
    parser.add_argument('--num', '-n', type=int)
    parser.add_argument('--infile', '-i', type=argparse.FileType(),
        help='optional file listing experiments (.exp) to run')
    parser.add_argument('--outfile', '-o', type=argparse.FileType('w'),
        help=('output file to dump list of bundles (.explist)'
              ', default: <dir>/bundles'))

    args = parser.parse_args()

    if not args.num:
        args.num = 7500
    elif not (0 < args.num <= 7500):
        parser.error('--num must be in [1..7500].')

    if not args.outfile:
        args.outfile = open(join(args.dir, 'bundles'), 'w')

    return args


def get_experiments_from_file(stream):
    experiments = stream.read().splitlines(keepends=True)
    stream.close()
    return experiments


def get_remaining_experiments(directory):
    exp_iter = glob.iglob('{}/working/*.exp'.format(directory))
    json_iter = glob.iglob('{}/working/*.json'.format(directory))
    all_exp = set(splitext(f)[0] for f in exp_iter)
    all_json = set(splitext(f)[0] for f in json_iter)
    return natsorted(f + '.exp\n' for f in all_exp - all_json)


def strided(l, n):
    sublists = [[] for i in range(n)]
    for i, item in enumerate(l):
        sublists[i % n].append(item)
    return sublists


def main():
    args = parse_args()

    if args.infile:
        experiments = get_experiments_from_file(args.infile)
    else:
        experiments = get_remaining_experiments(args.dir)

    bundles = strided(experiments, args.num)

    for i, bundle in enumerate(bundles):
        fname = join(args.dir, 'working', '{}.explist'.format(i))
        with open(fname, 'w') as f:
            f.writelines(bundle)
        args.outfile.write(fname + '\n')

    args.outfile.close()


if __name__ == '__main__':
    main()
