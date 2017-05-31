import argparse
import subprocess as sp
from os.path import (join, isfile, isdir, expanduser, realpath, normpath,
                     basename)


def main(args):
    ids = []
    script = expanduser('~/HMRI/code/boolnet/pbs/j_submit_single.sh')

    if not isfile(script):
        print('Error: must be run in pbs directory. Aborting.')
        return

    try:
        resources = 'walltime={},mem={}'.format(args.walltime, args.memory)
        for i in range(args.low_index, args.high_index + 1):
            cmd[2] = str(i)
            cmd = [script, args.dir, str(i), args.jobname, args.queue,
                   resources]
            status = sp.run(cmd, stdout=sp.PIPE, universal_newlines=True)
            ids.append(status.stdout)
    finally:
        print('\n'.join(str(s) for s in ids))


if __name__ == '__main__':
    # queues = ['computeq', 'xeon3q', 'xeon4q']
    queues = ['xeon3q']

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir', type=str)
    parser.add_argument('low_index', type=int)
    parser.add_argument('high_index', type=int)
    parser.add_argument('--queue', '-q', type=str, metavar='queue',
                        default='xeon3q', choices=queues)
    parser.add_argument('--jobname', '-n', type=str, default='SKF')
    parser.add_argument('--walltime', '-t', type=str, default='04:00:00')
    parser.add_argument('--memory', '-m', type=str, default='500mb')
    
    args = parser.parse_args()

    # Handle tilde
    args.dir = normpath(expanduser(args.dir))

    assert isdir(args.dir)
    assert args.low_index >= 0
    assert args.low_index <= args.high_index
    assert isfile(join(args.dir, 'working/{}.exp'.format(args.high_index)))
    assert len(args.jobname) <= 10

    main(args)
