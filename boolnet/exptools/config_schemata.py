import sys
import numpy as np
from good import (Schema, Required, In, All, Any, Range,
                  Type, IsDir, message, Optional)

from boolnet.bintools.functions import all_function_names
from boolnet.exptools.boolmapping import FileBoolMapping, OperatorBoolMapping


guiding_functions = all_function_names()


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

data_schema_generated = Schema({
    'type':                     'generated',
    'dir':                      IsDir(),
    'operator':                 str,
    'bits':                     All(int, Range(min=1)),
    Optional('out_width'):      All(int, Range(min=1)),
    Optional('window_size'):    All(int, Range(min=1))
    })

data_schema_file = Schema({
    'type':     'file',
    'dir':      IsDir(),
    'filename': str
    })

sampling_schema = Schema({
    'method':                   In({'given', 'generated'}),
    'Ns':                       All(int, Range(min=1)),
    'Ne':                       All(int, Range(min=1)),
    Optional('indices'):        [All(int, Range(min=0))],
    Optional('file_suffix'):    str
    })

network_schema_given = Schema({
    'method':           'given',
    'node_funcs':       'NAND',
    'file':             All(str, lambda v: v.endswith('.json')),
    'index':            All(int, Range(min=0)),
    'initial_gates':    All(Type(np.ndarray), is_2d, is_int_arr),
    })

network_schema_generated = Schema({
    'method':       'generated',
    'Ng':           All(int, Range(min=1)),
    'node_funcs':   Any(list, In(['NAND', 'NOR', 'random']))
    })

SA_schema = Schema({
    'name':                     'SA',
    'num_temps':                All(int, Range(min=1)),
    'init_temp':                Range(min=0.0),
    'temp_rate':                Range(min=0.0, max=1.0),
    'steps_per_temp':           All(int, Range(min=1)),
    'guiding_function':         In(guiding_functions),
    Optional('max_restarts'):   All(int, Range(min=0))
    })

HC_schema = Schema({
    'name':                     'HC',
    'max_iterations':           All(int, Range(min=1)),
    'guiding_function':         In(guiding_functions),
    Optional('max_restarts'):   All(int, Range(min=0))
    })

LAHC_schema = Schema({
    'name':                     'LAHC',
    'cost_list_length':         All(int, Range(min=1)),
    'max_iterations':           All(int, Range(min=1)),
    'guiding_function':         In(guiding_functions),
    Optional('max_restarts'):   All(int, Range(min=0))
    })


minFS_option_schema = Schema({
    'model':            In([1, 6]),
    'a_min':            All(int, Range(min=1)),
    'b_min':            All(int, Range(min=0)),
    Optional('cover'):  In(['alfa', 'beta', 'alfa+beta'])
    })


learner_schema_stratified = Schema({
    'name':                     'stratified',
    'optimiser':                Any(SA_schema, HC_schema, LAHC_schema),
    'inter_file_base':          str,
    Optional('kfs'):            bool,
    Optional('one_layer_kfs'):  bool,
    Optional('auto_target'):    bool,
    Optional('keep_files'):     bool,
    Optional('minfs_options'):  minFS_option_schema,
    })


learner_schema_basic = Schema({
    'name':                     'basic',
    'optimiser':                Any(SA_schema, HC_schema, LAHC_schema),
    })


learner_schema = Any(learner_schema_stratified,
                     learner_schema_basic)

network_schema = Schema(Any(network_schema_given,
                            network_schema_generated))

mapping_schema = Any(Type(FileBoolMapping),
                     Type(OperatorBoolMapping))

config_schema = Schema({
    'name':                     str,
    'data':                     Any(data_schema_file, data_schema_generated),
    'network':                  network_schema,
    'logging':                  In(['none', 'warning', 'info', 'debug']),
    'learner':                  learner_schema,
    'sampling':                 sampling_schema,
    'configuration_number':     All(int, Range(min=0)),
    'training_set_number':      All(int, Range(min=0)),
    'training_mapping':         mapping_schema,
    'test_mapping':             mapping_schema,
    Optional('seed'):           All(int, Range(0, sys.maxsize)),
    })
