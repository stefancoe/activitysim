# ActivitySim
# See full license in LICENSE.txt.

import os.path

import numpy.testing as npt
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pandana as pdna
import pytest


import orca

from .. import buffer
from .. import tracing


@pytest.fixture(scope='module')
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def configs_dir():
    return os.path.join(os.path.dirname(__file__), 'configs')


@pytest.fixture(scope='module')
def spec_name(data_dir):
    return os.path.join(data_dir, 'buffer_spec.csv')

@pytest.fixture(scope='module')
def net_name(data_dir):
    return os.path.join(data_dir, 'test_net.h5')

@pytest.fixture(scope='module')
def parcel_name(data_dir):
    return os.path.join(data_dir, 'test_parcels.csv')

@pytest.fixture(scope='module')
def data_name(data_dir):
    return os.path.join(data_dir, 'data.csv')


@pytest.fixture(scope='module')
def data(data_name):
    return pd.read_csv(data_name)


def test_read_model_spec(spec_name):

    spec = buffer.read_buffer_spec(spec_name)

    #assert len(spec) == 8

    assert list(spec.columns) == ['description', 'target', 'variable', 'target_df', 'expression']


def test_assign_variables(capsys, spec_name, net_name, parcel_name):

    spec = buffer.read_buffer_spec(spec_name)
  
    network = pdna.Network.from_hdf5(net_name)

    #network.init_pois(num_categories=constants["num_categories"], 
                      #max_dist=constants["max_dist"], max_pois=constants["max_pois"])

    parcel_data_df = pd.read_csv(parcel_name)
    assert len(parcel_data_df) == 2496

    # 
    parcel_data_df['node_id'] = network.get_node_ids(parcel_data_df['xcoord_p'].values,
                                                     parcel_data_df['ycoord_p'].values)

    # parcel 1219983 should be assigned to network node 82030 becuase it is the nearest node:
    assert int(parcel_data_df[parcel_data_df['parcelid']==1219983].node_id) == 82030

    locals_d = {
        'network': network,
        'parcels_df': parcel_data_df,
    }
    #parcel_data_df.set_index(constants['parcel_index'], inplace=True)
    #parcel_data_df.reset_index(level=0, inplace=True)

#    locals_d = {'CONSTANT': 7, '_shadow': 99}

#    results, trace_results, trace_assigned_locals \
#        = assign.assign_variables(spec, data, locals_d, trace_rows=None)

#    print results

#    assert list(results.columns) == ['target1', 'target2', 'target3']
#    assert list(results.target1) == [True, False, False]
#    assert list(results.target2) == [53, 53, 55]
#    assert list(results.target3) == [530, 530, 550]
#    assert trace_results is None
#    assert trace_assigned_locals is None

#    trace_rows = [False, True, False]

#    results, trace_results, trace_assigned_locals \
#        = assign.assign_variables(spec, data, locals_d, trace_rows=trace_rows)

#    # should get same results as before
#    assert list(results.target3) == [530, 530, 550]

#    # should assign trace_results for second row in data
#    print trace_results

#    assert trace_results is not None
#    assert '_scalar' in trace_results.columns
#    assert list(trace_results['_scalar']) == [42]

#    # shadow should have been assigned
#    assert list(trace_results['_shadow']) == [1]
#    assert list(trace_results['_temp']) == [9]
#    assert list(trace_results['target3']) == [530]

#    print "trace_assigned_locals", trace_assigned_locals
#    assert trace_assigned_locals['_DF_COL_NAME'] == 'thing2'

#    # shouldn't have been changed even though it was a target
#    assert locals_d['_shadow'] == 99

#    out, err = capsys.readouterr()


#def test_assign_variables_aliased(capsys, data):

#    spec_name = \
#        os.path.join(os.path.dirname(__file__), 'data', 'assignment_spec_alias_df.csv')

#    spec = assign.read_assignment_spec(spec_name)

#    locals_d = {'CONSTANT': 7, '_shadow': 99}

#    trace_rows = [False, True, False]

#    results, trace_results, trace_assigned_locals \
#        = assign.assign_variables(spec, data, locals_d,
#                                  df_alias='aliased_df', trace_rows=trace_rows)

#    print results

#    assert list(results.columns) == ['target1', 'target2', 'target3']
#    assert list(results.target1) == [True, False, False]
#    assert list(results.target2) == [53, 53, 55]
#    assert list(results.target3) == [530, 530, 550]

#    # should assign trace_results for second row in data
#    print trace_results

#    assert trace_results is not None
#    assert '_scalar' in trace_results.columns
#    assert list(trace_results['_scalar']) == [42]

#    # shadow should have been assigned
#    assert list(trace_results['_shadow']) == [1]
#    assert list(trace_results['_temp']) == [9]
#    assert list(trace_results['target3']) == [530]

#    assert locals_d['_shadow'] == 99

#    out, err = capsys.readouterr()


#def test_assign_variables_failing(capsys, data):

#    output_dir = os.path.join(os.path.dirname(__file__), 'output')
#    orca.add_injectable("output_dir", output_dir)

#    tracing.config_logger(basic=True)

#    spec_name = \
#        os.path.join(os.path.dirname(__file__), 'data', 'assignment_spec_failing.csv')

#    spec = assign.read_assignment_spec(spec_name)

#    locals_d = {
#        'CONSTANT': 7,
#        '_shadow': 99,
#        'log': np.log,
#    }

#    with pytest.raises(NameError) as excinfo:
#        results, trace_results = assign.assign_variables(spec, data, locals_d, trace_rows=None)

#    out, err = capsys.readouterr()
#    # don't consume output
#    print out

#    # numpy warning should appear in log but not raise error
#    assert 'invalid value encountered in log' in out

#    # undefined variable should raise error
#    assert "'undefined_variable' is not defined" in str(excinfo.value)