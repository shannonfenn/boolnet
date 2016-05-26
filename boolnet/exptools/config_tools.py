from copy import deepcopy
from collections import MutableMapping
from os import fsync
from os.path import join, splitext
from progress.bar import Bar
import voluptuous as vol
import numpy as np
import json

from boolnet.bintools.packing import BitPackedMatrix
from boolnet.bintools.operator_iterator import operator_from_name, num_operands
from boolnet.exptools.config_schemata import experiment_schema


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
            d[k] = u[k]
    return d


def load_dataset(settings):
    data_settings = settings['data']
    sampling_settings = settings['sampling']

    if data_settings['type'] == 'file':
        return file_instance(data_settings, sampling_settings)
    elif data_settings['type'] == 'generated':
        return generated_instance(data_settings, sampling_settings)
    else:
        raise ValueError('Invalid dataset type {}'.
                         format(data_settings['type']))


def load_samples(params, N, Ni):
    # load samples from file
    # prepare filename
    Ns = params['Ns']
    Ne = params['Ne']
    directory = params['dir']
    suffix = params.get('file_suffix', '')
    base_name = '{}_{}_{}{}.npy'.format(Ni, Ns, Ne, suffix)

    # load sample indices
    sample_filename = join(directory, base_name)
    training_indices = np.load(sample_filename)
    return training_indices


def file_instance(data_settings, sampling_settings):
    # load data set from file
    dataset_filename = join(data_settings['dir'],
                            data_settings['filename'])
    if splitext(dataset_filename)[-1] == '':
        dataset_filename += '.npz'
    with np.load(dataset_filename) as dataset:
        data = dataset['matrix']
        Ne = dataset['Ne']
        Ni = dataset['Ni']

    training_indices = load_samples(
        sampling_settings, data.shape[0], Ni)
    # build list of train/test set instances
    instances = [{
        'type': 'raw',
        'matrix': BitPackedMatrix(data, Ne, Ni),
        'training_indices': training_indices[i, :]
        } for i in range(training_indices.shape[0])]

    return instances


def generated_instance(data_settings, sampling_settings):
    Nb = data_settings['bits']
    op = operator_from_name(data_settings['operator'])

    Ni = num_operands(op) * Nb
    N = 2**Ni

    # by default the output width is the operand width
    No = data_settings.get('out_width', Nb)
    # default window size of 4 (arbitrary at this point)
    window_size = data_settings.get('window_size', 4)

    training_indices = load_samples(sampling_settings, N, Ni)
    Ns, Ne = training_indices.shape

    # Parameters
    # build list of train/test set instances
    instances = [{
        'type': 'operator',
        'operator': op,
        'Nb': Nb,
        'No': No,
        'window_size': window_size,
        'training_indices': training_indices[i, :]
        } for i in range(Ns)]
    return instances


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
        # msg = ('Experiment config {} invalid:\n'.format(config_num) +
        #        '  msg: {}\n'.format(err.message) +
        #        '  expected: {}\n'.format(err.expected) +
        #        '  provided: {}\n'.format(err.provided) +
        #        '  key path: {}\n'.format(err.path) +
        #        '  validator: {}\n'.format(err.validator) +
        #        'Config generation aborted.')
        msg = ('Experiment config {} invalid: {}\nerror: {}\npath: {}\n'
               '\nConfig generation aborted.').format(
            config_num + 1, msg, err.error_message, err.path)
        raise ValidationError(msg)


def list_experiment(sequence):
    # we expect a list of dicts
    schema = vol.All(vol.Schema([{}], extra=vol.ALLOW_EXTRA),
                     vol.Length(min=1))
    try:
        schema(sequence)
    except vol.MultipleInvalid:
        raise ValidationError(
            '\'configurations\' must be a sequence of mappings.')
    return sequence


def product_experiment(products):
    # we expect a pair of list
    schema = vol.All([vol.All(vol.Schema([{}], extra=vol.ALLOW_EXTRA),
                              vol.Length(min=1))],
                     vol.Length(min=2, max=2))
    try:
        schema(products)
    except vol.MultipleInvalid:
        raise ValidationError(
            '\'products\' must be a length 2 sequence of mappings.')

    # check there are no common keys
    keys_0 = set().union(*[d.keys() for d in products[0]])
    keys_1 = set().union(*[d.keys() for d in products[1]])
    if keys_0.isdisjoint(keys_1):
        # build merged mappings for each pair from products
        # no need to worry about key clashes since we have tested for them
        # thanks to PEP 448 this works in python 3.5 and is allegedly fast:
        #     https://gist.github.com/treyhunner/f35292e676efa0be1728
        return [[{**d1, **d2} for d2 in products[1]] for d1 in products[0]]
    else:
        raise ValidationError(
            '\'products\' has mappings with common keys: {}.'.format(
                keys_0.intersection(keys_1)))


def split_variables_from_base(settings):
    settings = deepcopy(settings)

    if 'configurations' in settings and 'product' in settings:
        raise ValidationError(
            'Only one of \'configurations\' and \'product\' may be present.')
    elif 'configurations' in settings:
        variable_sets = list_experiment(settings['configurations'])
        # no need to keep this sub-dict around
        settings.pop('configurations')
    elif 'product' in settings:
        variable_sets = product_experiment(settings['product'])

        # no need to keep this sub-dict around
        settings.pop('product')
    else:
        print(('Warning: no configuration list or product found, running as '
               'single configuration'))
        variable_sets = {}

    return variable_sets, settings


def generate_configurations(settings):
    # the configurations approach involves essentially having
    # a new settings dict for each configuration and updating
    # it with values in each dict in the configurations list

    variable_sets, base_settings = split_variables_from_base(settings)

    # Build up the configuration list
    configurations = []

    bar = Bar('Generating configurations', max=len(variable_sets),
              suffix='%(index)d/%(max)d : %(eta)ds')
    bar.update()
    try:
        for config_num, variables in enumerate(variable_sets):
            # keep each configuration isolated
            config_settings = deepcopy(base_settings)
            # update the settings dict with the values for this configuration
            update_nested(config_settings, variables)
            # check the given config is a valid experiment
            validate_schema(config_settings, experiment_schema,
                            config_num, variables)
            # record the config number for debugging
            config_settings['configuration_number'] = config_num
            # !!REMOVED!! load initial network from file if required
            # handle_initial_network(config_settings)
            # load the data for this configuration
            instances = load_dataset(config_settings)

            configurations.append((config_settings, instances))
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
        for config, instances in configurations:
            # samples may be optionally sub-indexed
            indices = get_config_indices(instances, config)
            # for each sample
            for i in indices:
                task = deepcopy(config)
                task['mapping'] = instances[i]
                task['training_set_number'] = i
                tasks.append(task)
            bar.next()
    finally:
        # clean up progress bar before printing anything else
        bar.finish()
    return tasks
