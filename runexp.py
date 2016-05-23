from datetime import datetime       # for date for result dir
from multiprocessing import Pool    # non-distributed parallellism
from time import time               # timing
from progress.bar import Bar        # progress indicators
import os                           # for mkdir
import os.path                      # for path manipulation
import yaml                         # for loading experiment files
import sys                          # for path, exit
import shutil                       # file copying
import logging                      # for logging, duh
import argparse                     # CLI
import itertools                    # imap and count
import scoop                        # for distributed parallellism

from boolnet.learning.learn_boolnet import learn_bool_net
import boolnet.exptools.config_tools as config_tools


def initialise_logging(settings, result_dir):
    ''' sets up logging with user defined level. '''
    level_dict = {'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG,
                  'none': logging.ERROR}
    try:
        log_level = level_dict[settings['logging_level'].lower()]
    except:
        log_level = logging.WARNING
    logging.basicConfig(filename=os.path.join(result_dir, 'log'),
                        level=log_level)


def initialise_notifications(args):
    if args.notify:
        try:
            import requests
            import instapush
            try:
                with open(args.ip_config) as f:
                    ip_settings = yaml.load(f, Loader=yaml.CSafeLoader)
                appid = ip_settings['appid']
                secret = ip_settings['secret']
                return instapush.App(appid=appid, secret=secret)
            except FileNotFoundError:
                print('Instapush config not found: {}'.format(args.ip_config))
            except yaml.YAMLError as err:
                print('Invalid instapush config: {}'.format(err))
            except requests.exceptions.RequestException as err:
                print('Failed to initialise notifications: {}'.format(err))
        except ImportError:
            print('Failed to import notification APIs: {}'.format(err))
        # disable notifications in the event of any errors
        print('Notifications disabled.\n')
        return None
    else:
        return None


def notify(notifier, settings, total_time, notes='none'):
    if notifier is not None:
        import requests
        try:
            message = {'name': str(settings['name']),
                       'time': str(total_time),
                       'num warnings': str(0),  # not implemented
                       'notes': str(notes)}
            ret = notifier.notify(event_name='experiment_complete',
                                  trackers=message)
            if ret.get('error', False):
                print('Notification error: {}'.format(ret))
        except requests.exceptions.RequestException as err:
            print('Failed to send notification: {}.'.format(err))


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment',
                        type=argparse.FileType('r'),
                        default='experiment.yaml',
                        help='experiment config filename.')
    parser.add_argument('--numprocs', '-n', metavar='N', type=int,
                        default=8, choices=range(0, 17),
                        help='how many parallel processes to use (give 0 for '
                             'scoop).')
    parser.add_argument('--notify', action='store_true',
                        help='enable push notifications.')
    parser.add_argument('--ip-config', type=str,
                        default='instapush.cfg',
                        help='instapush config file path (for notifications).')
    parser.add_argument('-d', '--data-dir', type=str, metavar='dir',
                        default='experiments/datasets/functions',
                        help='base directory for datasets.')
    parser.add_argument('-s', '--sample-dir', type=str, metavar='dir',
                        default='experiments/datasets/samples',
                        help='base directory for sampling files.')
    parser.add_argument('-r', '--result-dir', type=str, metavar='dir',
                        default='experiments/results',
                        help='directory to store results in (in own subdir).')

    return parser.parse_args()


def create_result_dir(base_dir, exp_name):
    ''' Generates a new timestamped directory for the results and copies
        the experiment script, experiment file and git hash into it.'''
    # make directory for the results
    result_dir = os.path.join(base_dir, '{}_{}_'.format(
        exp_name, datetime.now().strftime('%y-%m-%d')))

    # find the first number i such that 'Results_datetime_i' is not
    # already a directory and make that new directory
    result_dir += next(str(i) for i in itertools.count()
                       if not os.path.isdir(result_dir + str(i)))
    os.makedirs(result_dir)

    # return the directory name for the results to go in
    return result_dir


def initialise(args):
    ''' This is responsible for setting up the output directory,
        checking the git repo isn't dirty and setting up logging.'''
    # Check version
    if sys.version_info.major != 3:
        sys.exit("Requires python 3.")

    # load experiment file
    settings = yaml.load(args.experiment, Loader=yaml.CSafeLoader)

    settings['data']['dir'] = os.path.abspath(args.data_dir)

    settings['sampling']['dir'] = os.path.abspath(args.sample_dir)

    # create result directory
    result_dir = create_result_dir(args.result_dir, settings['name'])

    # copy experiment config into results directory
    shutil.copy(args.experiment.name, result_dir)

    # initialise logging
    initialise_logging(settings, result_dir)

    return settings, result_dir


def run_tasks(tasks, num_processes, out_stream):
    out_stream.write('[')
    if num_processes == 1:
        run_sequential(tasks, out_stream)
    elif num_processes < 1:
        run_scooped(tasks, out_stream)
    else:
        # Run the actual learning as a parallel process
        run_parallel(tasks, num_processes, out_stream)
    out_stream.write(']')


def run_parallel(tasks, num_processes, out_stream):
    ''' runs the given configurations '''
    with Pool(processes=num_processes) as pool:
        bar = Bar('Parallelised ({})'.format(num_processes),
                  max=len(tasks), suffix='%(index)d/%(max)d : %(elapsed)ds')
        bar.update()
        # uses unordered map to ensure results are dumped as soon as available
        for i, result in enumerate(pool.imap_unordered(learn_bool_net, tasks)):
            config_tools.dump_results_partial(result, out_stream, i == 0)
            bar.next()
        bar.finish()


def scoop_worker_wrapper(*args, **kwargs):
    import traceback
    try:
        # call to actual worker code
        return learn_bool_net(*args, **kwargs)
    except:
        type, value, tb = sys.exc_info()
        lines = traceback.format_exception(type, value, tb)
        print(''.join(lines))
        raise


def run_scooped(tasks, out_stream):
    ''' runs the given configurations '''
    suffix_fmt = 'completed: %(index)d/%(max)d | elapsed: %(elapsed)ds'
    bar = Bar('Scooped', max=len(tasks), suffix=suffix_fmt)
    bar.update()
    # uses unordered map to ensure results are dumped as soon as available
    for i, result in enumerate(scoop.futures.map_as_completed(
            scoop_worker_wrapper, tasks)):
        config_tools.dump_results_partial(result, out_stream, i == 0)
        bar.next()
    bar.finish()


def run_sequential(tasks, out_stream):
    ''' runs the given configurations '''
    bar = Bar('Sequential', max=len(tasks),
              suffix='%(index)d/%(max)d : %(elapsed)ds')
    bar.update()
    # map gives an iterator so results are dumped as soon as available
    for i, result in enumerate(map(learn_bool_net, tasks)):
        config_tools.dump_results_partial(result, out_stream, i == 0)
        bar.next()
    bar.finish()


# ############################## MAIN ####################################### #
def main():
    start_time = time()

    args = parse_arguments()

    notifier = initialise_notifications(args)

    settings, result_dir = initialise(args)

    print('Directories initialised.')
    print('Results in: ' + result_dir + '\n')

    # generate learning tasks
    try:
        configurations = config_tools.generate_configurations(settings)
        print('Done: {} configurations generated.'.format(len(configurations)))
        tasks = config_tools.generate_tasks(configurations)
        print('Done: {} tasks generated.\n'.format(len(tasks)))
    except config_tools.ValidationError as err:
        print(err)
        print('\nExperiment aborted.')
        print('Result directory will still exist: {}'.format(result_dir))
        return

    with open(os.path.join(result_dir, 'results.json'), 'w') as results_stream:
        # Run the actual learning as a parallel process
        run_tasks(tasks, args.numprocs, results_stream)

    total_time = time() - start_time

    print('Runs completed in {}s.'.format(total_time))

    notify(notifier, settings, total_time)

if __name__ == '__main__':
    main()
