import argparse
import re
import glob
import os.path
import subprocess as sp


def walltime_arg_type(s):
    if re.fullmatch('[0-9]*[0-9]:[0-9][0-9]:[0-9][0-9]|[0-9]+', s):
        return s
    else:
        msg = 'Invalid walltime: {}'.format(s)
        raise argparse.ArgumentTypeError(msg)


def directory_type(directory):
    # Handle tilde
    directory = os.path.abspath(os.path.expanduser(directory))
    if os.path.isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid path'.format(directory))


def submit(script, bundles, queue, walltime, joblistfile, dry):
    ids = []
    script = os.path.expanduser(script)

    if not os.path.isfile(script):
        print('Error: script does not exist. Aborting.')
        print('Bad script path: ' + script)
        return
    # pbs job limit
    if len(bundles) > 7500:
        print('Error: cannot submit {} jobs. Aborting.'.format(
            len(bundles)))
        return

    try:
        resources = 'walltime={}'.format(walltime)
        for i, expfile in enumerate(bundles):
            sout = '{}.sout'.format(expfile)
            serr = '{}.serr'.format(expfile)
            cmd = [script, expfile, sout, serr, queue, resources]
            if dry:
                print(' '.join(cmd))
            else:
                status = sp.run(cmd, stdout=sp.PIPE, universal_newlines=True)
                ids.append(status.stdout)
    finally:
        print('{} jobs submitted.'.format(len(ids)))
        if joblistfile:
            joblistfile.writelines(ids)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir', type=directory_type,
                        help='Directory containing .explist files')
    parser.add_argument('walltime', type=walltime_arg_type)
    parser.add_argument('--script', '-s', type=str,
                        default='~/HMRI/code/boolnet/pbs/j_submit_single.sh',
                        help='job script path. Default <boolnet>/pbs/j_submit_single.sh')
    parser.add_argument('--queue', '-q', type=str,
                        metavar='queue', default='xeon3q',
                        choices=['computeq', 'xeon3q', 'xeon4q'])
    parser.add_argument('--out', '-o', type=argparse.FileType('w'),
                        help='file to dump job ids. Default <dir>/jobids')
    parser.add_argument('--dry', action='store_true',
                        help='print resulting commands instead of executing.')
    args = parser.parse_args()

    if not args.out:
        args.out = open(os.path.join(args.dir, 'jobids'), 'w')

    bundles = glob.glob(os.path.join(args.dir, '*.explist'))

    submit(args.script, bundles, args.queue, args.walltime, args.out, args.dry)


if __name__ == '__main__':
    main()
