import argparse
import glob
import subprocess as sp
from os.path import (join, isfile, isdir, expanduser, realpath, normpath,
                     splitext)


def parse_args():
    # queues = ['computeq', 'xeon3q', 'xeon4q']
    queues = ['xeon3q']

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir', type=str)
    parser.add_argument('--range', '-r', type=int, nargs=2)
    parser.add_argument('--list', '-l', type=str,
                        help='list of files (absolute or relative to dir)')
    parser.add_argument('--queue', '-q', type=str, metavar='queue',
                        default='xeon3q', choices=queues)
    parser.add_argument('--jobname', '-n', type=str, default='SKF')
    parser.add_argument('--walltime', '-t', type=str, default='04:00:00')
    # parser.add_argument('--memory', '-m', type=str, default='500mb')

    args = parser.parse_args()

    # Handle tilde
    args.dir = normpath(expanduser(args.dir))

    if not isdir(args.dir):
        parser.error('{} is not a directory.'.format(args.dir))
    if len(args.jobname) > 10:
        parser.error('jobname ({}) must be 10 char or less.'.format(
            args.jobname))
    if args.range and args.list:
        parser.error('--range and --list are mutually exclusive.')
    if args.list and not isfile(args.list):
        parser.error('--list requires a filename.')
    if args.range and not(0 <= args.range[0] <= args.range[1]):
        parser.error('--range must satisfy 0 <= first <= second.')

    return args


def main(experiments, base_jobname, queue, walltime):
    ids = []
    script = expanduser('~/HMRI/code/boolnet/pbs/j_submit_single.sh')

    if not isfile(script):
        print('Error: must be run in pbs directory. Aborting.')
        return

    try:
        resources = 'walltime={}'.format(walltime)
        for i, expfile in enumerate(experiments):
            sout = splitext(expfile)[0] + '.sout'
            serr = splitext(expfile)[0] + '.serr'
            jobname = '{}_{}'.format(base_jobname, i)
            cmd = [script, expfile, jobname, sout, serr, queue, resources]
            status = sp.run(cmd, stdout=sp.PIPE, universal_newlines=True)
            ids.append(status.stdout)
    finally:
        print('\n'.join(str(s) for s in ids))


if __name__ == '__main__':

    args = parse_args()

    if args.list:
        with open(args.list, 'r') as f:
            experiments = [join(args.dir, l.strip()) for l in f]
    elif args.range:
        experiments = [join(args.dir, 'working', '{}.exp'.format(i))
                       for i in range(args.range[0], args.range[1] + 1)]
    else:
        experiments = glob.glob('{}/working/*.exp'.format(args.dir))

    main(experiments, args.jobname, args.queue, args.walltime)
