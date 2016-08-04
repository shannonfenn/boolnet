from copy import deepcopy
from collections import MutableMapping
from progress.bar import Bar
import voluptuous as vol
import numpy as np
import os
import json
import random

import boolnet.exptools.config_schemata as sch
import boolnet.bintools.packing as pk
import boolnet.bintools.operator_iterator as op


def get_seed(key):
    ''' Keeps a registry of seeds for each key, if given a new
        key get_seed() generates a new seed for that key, but if
        given an existing key it returns that seed. Allows any number
        of named seeds.'''
    if 'registry' not in get_seed.__dict__:
        # first call, create the registry
        get_seed.registry = {}
    if key not in get_seed.registry:
        # non-existant key, generate a seed
        random.seed()  # use default randomness source to get a seed
        get_seed.registry[key] = random.randint(1, 2**32-1)
    return get_seed.registry[key]


# to signal schema validation failure
# (with custom message formatting)
class ValidationError(Exception):
    pass


class ExperimentJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dump_results_partial(results, stream, first):
    if not first:
        stream.write(',')
    json.dump(results, stream, cls=ExperimentJSONEncoder)
    stream.write('\n')
    # ensure data is written to disk immediately
    stream.flush()
    os.fsync(stream.fileno())


def update_nested(d, u):
    ''' this updates a dict with another where the two may contain nested
        dicts themselves (or more generally nested mutable mappings). '''
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            r = update_nested(d.get(k, {}), v)
            d[k] = r
        else:
            # preference to second mapping if k exists in d
            d[k] = u[k]
    return d


def build_filename(params, extension):
    ''' complicated filename helper.'''
    filename = params['filename']
    location = params.get('dir', None)
    # if 'filename' is absolute then ignore 'dir'
    if location and not os.path.isabs(params['filename']):
        filename = os.path.join(location, params['filename'])
    # add extension if missing
    if os.path.splitext(filename)[-1] == '':
        filename += extension
    return filename


def load_samples(params, N, Ni):
    if params['type'] == 'given':
        training_indices = np.array(params['indices'], dtype=np.uintp)
    if params['type'] == 'file':
        filename = build_filename(params, '.npy')
        training_indices = np.load(filename)
    elif params['type'] == 'generated':
        # this provided seed allows us to generate the same set
        # of training indices across multiple configurations
        Ns = params['Ns']
        Ne = params['Ne']
        s = params['seed']
        if isinstance(s, str):
            # s is actually a name
            s = get_seed(s)
        random.seed(s)
        training_indices = np.array([
            random.sample(range(N), Ne) for i in range(Ns)])
    return training_indices


def load_dataset(settings):
    data_settings = settings['data']

    if data_settings['type'] == 'file':
        instance, N, Ni = file_instance(data_settings)
    elif data_settings['type'] == 'generated':
        instance, N, Ni = generated_instance(data_settings)

    training_indices = load_samples(settings['sampling'], N, Ni)

    return [{**instance, 'training_indices': ind} for ind in training_indices]


def file_instance(params):
    filename = build_filename(params, '.npz')
    with np.load(filename) as dataset:
        Mp = dataset['matrix']
        # Ne in the dataset is the full number of examples which we are
        # referring to herein as 'N' to differentiate from the sample size
        N = dataset['Ne']
        Ni = dataset['Ni']

    # build list of train/test set instances
    instance = {
        'type': 'raw',
        'matrix': pk.BitPackedMatrix(Mp, N, Ni)
        }

    return instance, N, Ni


def generated_instance(params):
    Nb = params['bits']
    operator = op.operator_from_name(params['operator'])

    instance = {
        'type': 'operator',
        'operator': operator,
        'Nb': Nb,
        'No': params.get('out_width', Nb),  # defaults to operand width
        'window_size': params.get('window_size', 4)  # arbitrary default
    }
    Ni = op.num_operands(operator) * Nb

    return instance, 2**Ni, Ni


# def handle_initial_network(settings):
#     net_settings = settings['learner']['network']
#     net_method = net_settings['method']
#     if net_method == 'given':
#         data_dir = settings['data']['dir']
#         filename = net_settings['file']
#         index = net_settings['index']
#         with open(join(data_dir, filename)) as f:
#             gates = np.array(json.load(f)[index], dtype=np.uint32)
#             net_settings['initial_gates'] = gates


def validate_schema(config, schema, config_num, msg):
    try:
        schema(config)
    except vol.MultipleInvalid as err:
        msg = ('Experiment instance {} invalid: {}\nerror: {}\npath: {}\n'
               '\nConfig generation aborted.').format(
            config_num + 1, err, err.error_message, err.path)
        raise ValidationError(msg)


def split_variables_from_base(settings):
    # configuration sub-dicts are popped
    try:
        variable_sets = settings['list']
    except KeyError:
        try:
            products = settings['product']
            # build merged mappings for each pair from products
            variable_sets = [update_nested(deepcopy(d1), d2)
                             for d2 in products[1]
                             for d1 in products[0]]
        except KeyError:
            print('Warning: no variable configuration found.\n')
            variable_sets = [{}]

    return variable_sets, settings['base_config']


def insert_default_log_keys(settings):
    defaults = [
        ['learner', True, ['learner', 'name']],
        ['sample_seed', True, ['sampling', 'seed']],
        ['learner_seed', True, ['learner', 'seed']],
        ['config_num', True, ['configuration_number']],
        ['trg_set_num', True, ['training_set_number']],
        ['tfs', True, ['learner', 'network', 'node_funcs']],
        ['guiding_function', True, ['learner', 'optimiser',
                                    'guiding_function']],
        ['given_tgt_order', True, ['learner', 'target_order']],
        ['fs_sel_method', False, ['learner', 'minfs_selection_method']],
        ['fs_masking', False, ['learner', 'minfs_masking']],
        ['opt_{}', False, ['learner', 'optimiser', '.*']],
    ]
    # give preference to user log_keys
    defaults.extend(settings['base_config'].get('log_keys', []))
    settings['base_config']['log_keys'] = defaults
    return settings


def generate_configurations(settings):
    # validate the given schema
    try:
        sch.experiment_schema(settings)
    except vol.MultipleInvalid as err:
        raise ValidationError(
            'Top-level config invalid: {}\nerror: {}\npath: {}'.format(
                err, err.error_message, err.path))

    # insert default log_keys values into base config
    insert_default_log_keys(settings)

    # the configurations approach involves having a multiple config dicts and
    # updating them with each element of the configurations list or product
    variable_sets, base_settings = split_variables_from_base(settings)

    # Build up the configuration list
    configurations = []

    bar = Bar('Generating configurations', max=len(variable_sets),
              suffix='%(index)d/%(max)d : %(eta)ds')
    bar.update()
    try:
        for config_num, variables in enumerate(variable_sets):
            # keep contexts isolated
            context = deepcopy(base_settings)
            # update the settings dict with the values for this configuration
            update_nested(context, variables)
            # check the given config is a valid experiment
            validate_schema(context, sch.instance_schema,
                            config_num, variables)
            # record the config number for debugging
            context['configuration_number'] = config_num
            # !!REMOVED!! load initial network from file if required
            # handle_initial_network(context)
            # load the data for this configuration
            instances = load_dataset(context)

            configurations.append((context, instances))
            bar.next()
    finally:
        # clean up progress bar before printing anything else
        bar.finish()
    return configurations


def generate_tasks(configurations):
    # Build up the task list
    tasks = []

    bar = Bar('Generating training tasks', max=len(configurations),
              suffix='%(index)d/%(max)d : %(eta)ds')
    bar.update()
    try:
        for context, instances in configurations:
            # for each sample
            for i, instance in enumerate(instances):
                task = deepcopy(context)
                task['mapping'] = instance
                task['training_set_number'] = i
                tasks.append(task)
            bar.next()
    finally:
        # clean up progress bar before printing anything else
        bar.finish()
    return tasks
