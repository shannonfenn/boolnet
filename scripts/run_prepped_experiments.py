from datetime import datetime       # for date for result dir
from datetime import timedelta       # for date for result dir
from multiprocessing import Pool    # non-distributed parallellism
from time import time               # timing
from glob import glob
import yaml                         # for loading experiment files
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader
import sys                          # for path, exit
import argparse                     # CLI
import pickle
import json
import gzip
import itertools
from os.path import join, isfile, isdir, expanduser, normpath, splitext, exists

from boolnet.utils import BetterETABar
from boolnet.exptools.learn_boolnet import learn_bool_net
from boolnet.utils import NumpyAwareJSONEncoder


def initialise_notifications(args):
    if args.email:
        config_path = expanduser(args.email_config)
        try:
            with open(config_path) as f:
                settings = yaml.load(f, Loader=Loader)
            return settings
        except FileNotFoundError:
            print('Email config not found: {}'.format(config_path))
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
                        default='~/email.cfg', help='email config file path.')
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


def run_tasks(task_iterator, stream, num_tasks):
    ''' Runs the given tasks and dumps to a single json file.
        Use num_tasks=0 for batch mode'''
    if num_tasks:
        bar = BetterETABar('Progress', max=num_tasks)
        bar.update()
    # map gives an iterator so results are dumped as soon as available
    for i, result in enumerate(task_iterator):
        stream.write('[' if i == 0 else '\n,')
        json.dump(result, stream, cls=NumpyAwareJSONEncoder,
                  separators=(',', ':'))
        if num_tasks:
            bar.next()
    stream.write('\n]\n')
    if num_tasks:
        bar.finish()


def run_single_experiment(expfile):
    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))
    result = learn_bool_net(task)
    result['id'] = task['id']
    return result


def consecutive_filename(base, file_fmt_str):
    # find first path matching 'base/*i*' which doesn't exist
    fmt_str = join(expanduser(base), file_fmt_str)
    return next(fmt_str.format(i) for i in itertools.count()
                if not exists(fmt_str.format(i)))


# ############################## MAIN ####################################### #
def main():
    start_time = time()

    args = parse_arguments()

    notifier = initialise_notifications(args)

    if args.list:
        with open(args.list, 'r') as f:
            tasks = [join(args.dir, l.strip()) for l in f]
    elif args.range:
        tasks = [join(args.dir, 'tasks', '{}.exp'.format(i))
                 for i in range(args.range[0], args.range[1] + 1)]
    else:
        tasks = glob(join(args.dir, 'tasks', '*.exp'))

    resultfile = consecutive_filename(args.dir, 'results_raw_{}.json')

    print('{} unprocessed .exp files found.'.format(len(tasks)))
    print('Results in: {}.'.format(resultfile))

    if len(tasks) > 0:
        with open(resultfile, 'w') as resultstream:
            if args.numprocs < 1:
                print('Dispatching with SCOOP.')
                import scoop
                task_iterator = scoop.futures.map_as_completed(
                    scoop_worker_wrapper, tasks)
                run_tasks(task_iterator, resultstream,
                          0 if args.batch_mode else len(tasks))
            elif args.numprocs == 1:
                print('Dispatching with map.')
                task_iterator = map(run_single_experiment, tasks)
                run_tasks(task_iterator, resultstream,
                          0 if args.batch_mode else len(tasks))
            else:
                print('Dispatching with multiprocessing.')
                # Run the actual learning as a parallel process
                with Pool(processes=args.numprocs) as pool:
                    task_iterator = pool.imap_unordered(
                        run_single_experiment, tasks)
                    run_tasks(task_iterator, resultstream,
                              0 if args.batch_mode else len(tasks))

    total_time = time() - start_time

    notify(notifier, args.dir, total_time)


if __name__ == '__main__':
    main()
