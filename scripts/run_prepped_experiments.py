from datetime import datetime       # for date for result dir
from datetime import timedelta       # for date for result dir
from multiprocessing import Pool    # non-distributed parallellism
from time import time               # timing
import yaml                         # for loading experiment files
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader
import sys                          # for path, exit
import argparse                     # CLI
import glob
import pickle
import json
import gzip
from os.path import join, isfile, isdir, expanduser, normpath, splitext

from boolnet.utils import BetterETABar
from boolnet.exptools.learn_boolnet import learn_bool_net
from boolnet.utils import NumpyAwareJSONEncoder


def initialise_notifications(args):
    if args.email:
        try:
            with open(args.email_config) as f:
                settings = yaml.load(f, Loader=Loader)
            return settings
        except FileNotFoundError:
            print('Email config not found: {}'.format(args.email_config))
        except yaml.YAMLError as err:
            print('Invalid email config: {}'.format(err))
        # disable notifications in the event of any errors
        print('Notifications disabled.\n')
        return None
    else:
        return None


def notify(notifier, name, runtime):
    if notifier:
        import smtplib
        fromaddr = notifier['fromaddr']
        toaddr = notifier['toaddr']
        # init SMTP server
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        server.login(notifier['usr'], notifier['psw'])
        # compose email
        date = datetime.strftime(datetime.now(), '%Y-%m-%d')
        runtime = str(timedelta(seconds=runtime))
        subject = 'Experiment completed'
        header = ('Date: {}\r\nFrom: {}\r\nTo: {}\r\nSubject: {}\r\nX-Mailer: '
                  'My-Mail\r\n\r\n').format(date, fromaddr, toaddr, subject)
        body = 'name: {}\nruntime: {}'.format(name, runtime)
        # send
        server.sendmail(fromaddr, toaddr, header+body)
        server.quit()


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dir', type=str,
                        help='experiment root directory.')
    parser.add_argument('-n', '--numprocs', metavar='N', type=int, default=8,
                        help='how many parallel processes to use (give 0 for '
                             'scoop).')
    parser.add_argument('-e', '--email', action='store_true',
                        help='enable email notifications.')
    parser.add_argument('-c', '--email-config', metavar='file', type=str,
                        default='email.cfg', help='email config file path.')
    parser.add_argument('-b', '--batch-mode', action='store_true',
                        help='suppress progress bars.')
    parser.add_argument('--range', '-r', type=int, nargs=2)
    parser.add_argument('--list', '-l', type=str,
                        help='list of files (absolute or relative to dir)')

    args = parser.parse_args()

    # Handle tilde
    args.dir = normpath(expanduser(args.dir))

    if not isdir(args.dir):
        parser.error('{} is not a directory.'.format(args.dir))
    if args.range and args.list:
        parser.error('--range and --list are mutually exclusive.')
    if args.list and not isfile(args.list):
        parser.error('--list requires a filename.')
    if args.range and not(0 <= args.range[0] <= args.range[1]):
        parser.error('--range must satisfy 0 <= first <= second.')

    return args


def get_remaining_experiments(directory):
    all_exp = set(splitext(f)[0] for f in glob.iglob(
        '{}/working/*.exp'.format(directory)))
    all_json = set(splitext(f)[0] for f in glob.iglob(
        '{}/working/*.json'.format(directory)))
    remaining = all_exp - all_json
    return [f + '.exp' for f in remaining]


def scoop_worker_wrapper(*args, **kwargs):
    import traceback
    try:
        # call to actual worker code
        return run_single_experiment(*args, **kwargs)
    except:
        type, value, tb = sys.exc_info()
        lines = traceback.format_exception(type, value, tb)
        print(''.join(lines))
        raise


def run_scooped(expfiles, batch_mode):
    import scoop
    ''' runs the given configurations '''
    if not batch_mode:
        bar = BetterETABar('Scooped', max=len(expfiles))
        bar.update()
    # uses unordered map to ensure results are dumped as soon as available
    for _ in scoop.futures.map_as_completed(scoop_worker_wrapper, expfiles):
        if not batch_mode:
            bar.next()
    if not batch_mode:
        bar.finish()


def run_sequential(expfiles, batch_mode):
    ''' runs the given configurations '''
    if not batch_mode:
        bar = BetterETABar('Sequential', max=len(expfiles))
        bar.update()
    # map gives an iterator so results are dumped as soon as available
    for _ in map(run_single_experiment, expfiles):
        if not batch_mode:
            bar.next()
    if not batch_mode:
        bar.finish()


def run_parallel(expfiles, num_processes, batch_mode):
    ''' runs the given configurations '''
    with Pool(processes=num_processes) as pool:
        if not batch_mode:
            barname = 'Parallelised ({})'.format(num_processes)
            bar = BetterETABar(barname, max=len(expfiles))
            bar.update()
        # uses unordered map to ensure results are dumped as soon as available
        for _ in pool.imap_unordered(run_single_experiment, expfiles):
            if not batch_mode:
                bar.next()
        if not batch_mode:
            bar.finish()


def run_single_experiment(expfile):
    resultfile = splitext(expfile)[0] + '.json'

    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))

    result = learn_bool_net(task)

    with open(resultfile, 'w') as stream:
        json.dump(result, stream, cls=NumpyAwareJSONEncoder)


# ############################## MAIN ####################################### #
def main():
    start_time = time()

    args = parse_arguments()

    notifier = initialise_notifications(args)

    if args.list:
        with open(args.list, 'r') as f:
            experiments = [join(args.dir, l.strip()) for l in f]
    elif args.range:
        experiments = [join(args.dir, 'working', '{}.exp'.format(i))
                       for i in range(args.range[0], args.range[1] + 1)]
    else:
        experiments = get_remaining_experiments(args.dir)

    print('{} unprocessed .exp files found.'.format(len(experiments)))

    # Run the actual learning as a parallel process
    if args.numprocs == 1:
        run_sequential(experiments, args.batch_mode)
    elif args.numprocs < 1:
        run_scooped(experiments, args.batch_mode)
    else:
        # Run the actual learning as a parallel process
        run_parallel(experiments, args.numprocs, args.batch_mode)

    total_time = time() - start_time

    notify(notifier, args.dir, total_time)


if __name__ == '__main__':
    main()
