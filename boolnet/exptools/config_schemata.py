import sys
import numpy as np
from good import Schema, Required, In, All, Any, Range, Type, IsDir
from good import message, Allow, Optional

from boolnet.bintools.metrics import all_metric_names
from boolnet.exptools.boolmapping import FileBoolMapping, OperatorBoolMapping
from boolnet.learning.learn_boolnet import OPTIMISERS


metrics = all_metric_names()


@message('2D array expected')
def is_2d(v):
    if len(v.shape) != 2:
        raise ValueError
    return v


@message('1D integer array expected')
def is_1d(v):
    if len(v.shape) != 1:
        raise ValueError
    return v


@message('Integer array expected')
def is_int_arr(v):
    if not np.issubdtype(v.dtype, np.integer):
        raise ValueError
    return v

SA_schema = Schema({
    'name':           'SA',
    'metric':         In(metrics),
    'num_temps':      All(int, Range(min=1)),
    'init_temp':      Range(min=0.0),
    'temp_rate':      Range(min=0.0, max=1.0),
    'steps_per_temp': All(int, Range(min=1))
    }, default_keys=Required)

SA_VN_schema = Schema({
    'name':             'SA-VN',
    'metric':           In(metrics),
    'num_temps':        All(int, Range(min=1)),
    'init_temp':        Range(min=0.0),
    'temp_rate':        Range(min=0.0, max=1.0),
    'steps_per_temp':   All(int, Range(min=1)),
    'init_move_count':  All(int, Range(min=1))
    }, default_keys=Required)

LAHC_schema = Schema({
    'name':             'LAHC',
    'metric':           In(metrics),
    'cost_list_length': All(int, Range(min=1)),
    'max_iterations':   All(int, Range(min=1))
    }, default_keys=Required)

LAHC_VN_schema = Schema({
    'name':             'LAHC-VN',
    'metric':           In(metrics),
    'cost_list_length': All(int, Range(min=1)),
    'max_iterations':   All(int, Range(min=1)),
    'init_move_count':  All(int, Range(min=1))
    }, default_keys=Required)

TS_schema = Schema({
    'name':             'TS',
    'metric':           In(metrics),
    'tabu_period':      All(int, Range(min=1)),
    'max_iterations':   All(int, Range(min=1))
    }, default_keys=Required)

optimiser_name_schema = Schema({
    'name':     In(OPTIMISERS.keys())
    }, extra_keys=Allow)

optimiser_schema = Schema(
    All(Any(SA_schema, SA_VN_schema, LAHC_schema, LAHC_VN_schema, TS_schema),
        optimiser_name_schema))

data_schema_generated = Schema({
    'type':                     'generated',
    'dir':                      IsDir(),
    'operator':                 str,
    'bits':                     All(int, Range(min=1)),
    Optional('No'):             All(int, Range(min=1)),
    Optional('window_size'):    All(int, Range(min=1))
    }, default_keys=Required)

data_schema_file = Schema({
    'type':     'file',
    'dir':      IsDir(),
    'filename': str
    }, default_keys=Required)

sampling_schema = Schema({
    'method':                   In({'given', 'generated'}),
    'Ns':                       All(int, Range(min=1)),
    'Ne':                       All(int, Range(min=1)),
    Optional('indices'):        [All(int, Range(min=0))],
    Optional('file_suffix'):    str
    }, default_keys=Required)

network_schema_given = Schema({
    'method':           'given',
    'node_funcs':       'NAND',
    'file':             All(str, lambda v: v.endswith('.json')),
    'index':            All(int, Range(min=0)),
    'initial_gates':    All(Type(np.ndarray), is_2d, is_int_arr),
    }, default_keys=Required)

network_schema_generated = Schema({
    'method':       'generated',
    'Ng':           All(int, Range(min=1)),
    'node_funcs':   In(['NAND', 'NOR', 'random'])
    }, default_keys=Required)


learner_schema = Schema({
    'name':     In(['basic', 'stratified']),
    Optional('kfs'):                bool,
    Optional('feature_masking'):    bool,
    Optional('smaller_kfs_inputs'): bool
    })

network_schema = Schema(Any(network_schema_given,
                            network_schema_generated))

config_schema = Schema({
    'name':                     str,
    'data':                     Any(data_schema_file, data_schema_generated),
    'network':                  network_schema,
    'logging':                  In(['none', 'warning', 'info', 'debug']),
    'learner':                  learner_schema,
    'optimiser':                optimiser_schema,
    'sampling':                 sampling_schema,
    'configuration_number':     All(int, Range(min=0)),
    'training_set_number':      All(int, Range(min=0)),
    'inter_file_base':          str,
    'training_mapping':         Any(Type(FileBoolMapping), Type(OperatorBoolMapping)),
    'test_mapping':             Any(Type(FileBoolMapping), Type(OperatorBoolMapping)),
    Optional('seed'):           All(int, Range(0, sys.maxsize)),
    Optional('fabcpp_options'):    list,
    }, default_keys=Required)
