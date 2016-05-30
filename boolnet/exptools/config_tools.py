from copy import deepcopy
from collections import MutableMapping
from os import fsync
from os.path import join, splitext
from progress.bar import Bar
import voluptuous as vol
import numpy as np
import json
import random

import boolnet.exptools.config_schemata as sch
import boolnet.bintools.packing as pk
import boolnet.bintools.operator_iterator as op


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
    fsync(stream.fileno())


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


def load_samples(params, N, Ni):
    Ns = params['Ns']
    Ne = params['Ne']
    if params['type'] == 'file':
        training_indices = np.load(params['file_name'])
    elif params['type'] == 'generated':
        # this provided seed allows us to generate the same set
        # of training indices across multiple configurations
        seed = params['seed']
        np.random.seed(seed)
        training_indices = np.random.choice(N, size=(Ns, Ne), replace=False)
    return training_indices


def load_dataset(settings):
    data_settings = settings['data']

    if data_settings['type'] == 'file':
        instance, N, Ni = file_instance(data_settings)
    elif data_settings['type'] == 'generated':
        instance, N, Ni = generated_instance(data_settings)

    training_indices = load_samples(settings['sampling'], N, Ni)

    return [{**instance, 'training_indices': ind} for ind in training_indices]


def file_instance(data_settings):
    # load data set from file
    dataset_filename = join(data_settings['dir'],
                            data_settings['filename'])
    if splitext(dataset_filename)[-1] == '':
        dataset_filename += '.npz'
    with np.load(dataset_filename) as dataset:
        data = dataset['matrix']
        Ne = dataset['Ne']
        Ni = dataset['Ni']

    # build list of train/test set instances
    instance = {
        'type': 'raw',
        'matrix': pk.BitPackedMatrix(data, Ne, Ni)
        }

    return instance, data.shape[0], Ni


def generated_instance(data_settings):
    Nb = data_settings['bits']
    operator = op.operator_from_name(data_settings['operator'])

    instance = {
        'type': 'operator',
        'operator': operator,
        'Nb': Nb,
        'No': data_settings.get('out_width', Nb),  # defaults to operand width
        'window_size': data_settings.get('window_size', 4)  # arbitrary default
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


def get_config_indices(instances, config_settings):
    # samples may be optionally sub-indexed
    if 'indices' in config_settings['sampling']:
        config_indices = config_settings['sampling']['indices']
        if any(i >= len(instances) for i in config_indices):
            raise ValueError('sampling indices has elements larger than Ns')
    else:
        config_indices = range(len(instances))
    return config_indices


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
            print('Warning: no variable configuration found.')
            variable_sets = [{}]

    return variable_sets, settings['base_config']


def update_seeding(settings):
    for context in ['sampling', 'learner']:
        seed = settings['seeding'][context]
        if seed == 'shared':
            random.seed()  # use default randomness source to get a seed
            seed = random.randint(1, 2**32-1)
        elif seed == 'unique':
            seed = None  # this will cause seed() to reload for each instance
        settings['base_config'][context]['seed'] = seed


def generate_configurations(settings, data_dir, sampling_dir):
    # validate the given schema
    try:
        sch.experiment_schema(settings)
    except vol.MultipleInvalid as err:
        raise ValidationError(
            'Top-level config invalid: {}\nerror: {}\npath: {}'.format(
                err, err.error_message, err.path))

    # insert given values into base config
    settings['base_config']['data']['dir'] = data_dir
    settings['base_config']['sampling']['dir'] = sampling_dir
    # insert seed values into base config
    update_seeding(settings)

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
            # samples may be optionally sub-indexed
            indices = get_config_indices(instances, context)
            # for each sample
            for i in indices:
                task = deepcopy(context)
                task['mapping'] = instances[i]
                task['training_set_number'] = i
                tasks.append(task)
            bar.next()
    finally:
        # clean up progress bar before printing anything else
        bar.finish()
    return tasks
