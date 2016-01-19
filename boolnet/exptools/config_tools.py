from copy import deepcopy
from collections import MutableMapping, namedtuple
from os import fsync
from os.path import join, splitext
from progress.bar import Bar
import numpy as np
import json

from boolnet.bintools.operator_iterator import operator_from_name, num_operands
from boolnet.exptools.boolmapping import BoolMapping, OperatorBoolMapping
from boolnet.exptools.config_schemata import config_schema


Instance = namedtuple('Instance', ['mapping', 'training_indices'])


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


def load_samples(params, data_dir, N, Ni):
    # load samples from file
    # prepare filename
    Ns = params['Ns']
    Ne = params['Ne']
    suffix = params.get('file_suffix', '')
    base_name = '{}_{}_{}{}.npy'.format(Ni, Ns, Ne, suffix)

    # load sample indices
    sample_filename = join(data_dir, 'samples', base_name)
    training_indices = np.load(sample_filename)
    return training_indices


def pack_examples(data, training_indices):
    ''' Parititions the given function into training and test sets,
        based on the given training indices.'''
    # Parameters
    N, Ni = data.shape[0]
    Ns, Ne = training_indices.shape
    # Generate the test indices array, each row should contain all
    # indices not in the equivalent row of training_indices
    test_indices = np.zeros(shape=(Ns, N - Ne), dtype=int)
    for s in range(Ns):
        test_indices[s] = np.setdiff1d(np.arange(N), training_indices[s])
    # Using numpy's advanced indexing we can get the sets
    train_inps = inputs[training_indices]
    train_tgts = targets[training_indices]
    test_inps = inputs[test_indices]
    test_tgts = targets[test_indices]

    # build list of train/test set instances
    instances = [Instance(
        mapping=data,
        Ni=Ni,
        training_indices=training_indices[i]

        training_mapping=BoolMapping(train_inps[i], train_tgts[i], Ne),
        test_mapping=BoolMapping(test_inps[i], test_tgts[i], N - Ne),
        training_indices=training_indices[i]
        ) for i in range(Ns)]

    return instances


def file_instance(data_settings, sampling_settings):
    data_dir = data_settings['dir']
    # load data set from file
    dataset_filename = join(data_dir, 'functions', data_settings['filename'])
    if splitext(dataset_filename)[-1] == '':
        dataset_filename += '.npz'
    with np.load(dataset_filename) as dataset:
        data = dataset['matrix']
        Ni = dataset['Ni']

    N, Ni = data.shape
    training_indices = load_samples(sampling_settings, data_dir, N, Ni)
    # partition the sets based on loaded indices
    return pack_examples(inputs, targets, training_indices)


def generated_instance(data_settings, sampling_settings):
    data_dir = data_settings['dir']
    Nb = data_settings['bits']
    op = operator_from_name(data_settings['operator'])

    Ni = num_operands(op) * Nb
    N = 2**Ni

    # by default the output width is the operand width
    No = data_settings.get('out_width', Nb)
    # default window size of 4 (arbitrary at this point)
    window_size = data_settings.get('window_size', 4)

    training_indices = load_samples(sampling_settings, data_dir, N, Ni)
    Ns, Ne = training_indices.shape

    # Parameters
    # build list of train/test set instances
    instances = [Instance(
        training_mapping=OperatorBoolMapping(
            training_indices[i], Nb, Ni, No, window_size, op, 0),
        test_mapping=OperatorBoolMapping(
            training_indices[i], Nb, Ni, No, window_size, op, N),
        training_indices=training_indices[i]
        ) for i in range(Ns)]
    return instances


def handle_initial_network(settings):
    net_method = settings['network']['method']
    if net_method == 'given':
        data_dir = settings['data']['dir']
        filename = settings['network']['file']
        index = settings['network']['index']
        with open(join(data_dir, filename)) as f:
            gates = np.array(json.load(f)[index], dtype=np.uint32)
            settings['network']['initial_gates'] = gates


def get_config_indices(instances, config_settings):
    # samples may be optionally sub-indexed
    if 'indices' in config_settings['sampling']:
        config_indices = config_settings['sampling']['indices']
        if any(i >= len(instances) for i in config_indices):
            raise ValueError('sampling indices has elements larger than Ns')
    else:
        config_indices = range(len(instances))
    return config_indices


def generate_configurations(settings):
    # the configurations approach involves essentially having
    # a new settings dict for each configuration and updating
    # it with values in each dict in the configurations list
    settings = deepcopy(settings)

    variable_sets = settings['configurations']
    # no need to keep this sub-dict around
    settings.pop('configurations')
    # Build up the configuration list
    configurations = []

    bar = Bar('Generating configurations', max=len(variable_sets),
              suffix='%(index)d/%(max)d : %(eta)ds')
    bar.update()
    for config_no, variables in enumerate(variable_sets):
        # keep each configuration isolated
        config_settings = deepcopy(settings)
        # update the settings dict with the values for this configuration
        update_nested(config_settings, variables)

        config_schema(config_settings)

        # record the config number for debugging
        config_settings['configuration_number'] = config_no
        # load initial network from file if required
        handle_initial_network(config_settings)
        # load the data for this configuration
        instances = load_dataset(config_settings)

        configurations.append((config_settings, instances))
        bar.next()
    bar.finish()
    return configurations


def generate_tasks(configurations):
    # Build up the task list
    tasks = []

    bar = Bar('Generating training tasks', max=len(configurations),
              suffix='%(index)d/%(max)d : %(eta)ds')
    bar.update()
    for config, instances in configurations:
        # samples may be optionally sub-indexed
        indices = get_config_indices(instances, config)

        conf_num = config['configuration_number']
        # for each training set
        for i in indices:
            instance = instances[i]

            task = deepcopy(config)
            task['training_mapping'] = instance.training_mapping
            task['test_mapping'] = instance.test_mapping
            task['training_indices'] = instance.training_indices
            task['training_set_number'] = i
            task['learner']['inter_file_base'] += '{}_{}_'.format(conf_num, i)

            # dump the iteration settings out
            tasks.append(task)
        bar.next()
    bar.finish()
    return tasks
