#! /usr/bin/env python

import yaml                         # for loading experiment files
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader
import argparse                     # CLI
from good import Invalid
import boolnet.exptools.config_tools as cft
import boolnet.exptools.config_schemata as sch


def check(settings, full=False):
    # validate the given schema
    try:
        sch.experiment_schema(settings)
    except Invalid as err:
        raise cft.ValidationError(
            'Top-level config invalid: {}'.format(err))
    seed_handler = cft.SeedHandler(settings['seed'])
    # insert default log_keys values into base config
    cft.insert_default_log_keys(settings)
    # the configurations approach involves having a multiple config dicts and
    # updating them with each element of the configurations list or product
    variable_sets, base_settings = cft.split_variables_from_base(settings)

    configurations = cft._generate_configurations(
        variable_sets, base_settings, seed_handler, False)
    print('{} configurations.'.format(len(configurations)))
    if full:
        tasks = cft._generate_tasks(configurations, seed_handler, False)
        print('{} tasks.'.format(len(tasks)))


def main():
    parser = argparse.ArgumentParser(description='Config file tester - to help'
                                                 ' debug config file problems '
                                                 'w/o running experiments.')

    parser.add_argument('experiment',
                        type=argparse.FileType('r'),
                        help='experiment config filename.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='generate tasks as well.')

    args = parser.parse_args()

    # load experiment file
    settings = yaml.load(args.experiment, Loader=Loader)

    # test generation of tasks
    try:
        check(settings, args.verbose)
        print('\nExperiment config is valid.')
    except cft.ValidationError as err:
        print()
        print(err)
        print('\nExperiment config is NOT valid.')
        return


if __name__ == '__main__':
    main()
