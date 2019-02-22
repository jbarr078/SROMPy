import pytest
import os
import sys
import numpy as np

if 'PYTHONPATH' not in os.environ:
    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

sys.path.insert(0, '/home/jmbarrie/Documents/DevOps/SROMPy/tests/test_scripts_data')

from SROMPy.target import BetaRandomVariable, SampleRandomVector
from SROMPy.srom import SROM, SROMSimulator, SROMSurrogate, \
                        FiniteDifference as FD
from spring_mass_model import SpringMassModel

@pytest.fixture
def beta_random_variable():
    random_input = \
        BetaRandomVariable(alpha=3.0, beta=2.0, shift=1.0, scale=2.5)

    return random_input

@pytest.fixture
def spring_model_fixture():
    spring_model = SpringMassModel(state0=[0.0, 0.0], time_step=0.01)

    return spring_model

@pytest.fixture
def srom_simulator_fixture():
    random_variable = \
        BetaRandomVariable(alpha=3.0, beta=2.0, shift=1.0, scale= 2.5)

    spring_model = SpringMassModel(state0=[0.0, 0.0], time_step=0.01)

    srom_sim = SROMSimulator(random_variable, spring_model)
    return srom_sim

@pytest.fixture
def srom_base_fixture(beta_random_variable):
    srom = SROM(size=10, dim=1)
    srom.optimize(beta_random_variable)

    return srom

def test_simulator_init_exception_for_invalid_parameters(beta_random_variable, 
                                                         spring_model_fixture):

    with pytest.raises(TypeError):
        SROMSimulator(1, spring_model_fixture)

    with pytest.raises(TypeError):
        SROMSimulator(beta_random_variable, 'Not A Proper Model')

def test_simulate_exception_for_invalid_parameters(srom_simulator_fixture):
    with pytest.raises(TypeError):
        srom_simulator_fixture.simulate(10.5, 'PWC', 1e-12)

    with pytest.raises(ValueError):
        srom_simulator_fixture.simulate(10, 'no', 1e-12)

    with pytest.raises(TypeError):
        srom_simulator_fixture.simulate(10, 'PWL')


def test_simulate_pwc_spring_mass(srom_simulator_fixture):
    '''
    Tests a PWC surrogate against a manual reference solution generated from  
    test_scripts_data/generate_srom_sim_ref_solution.py
    '''

    pwc_surrogate = srom_simulator_fixture.simulate(10, 'PWC')

    mean_pwc = pwc_surrogate.compute_moments(1)[0][0]
    mean_reference = 12.385393457327542

    assert np.isclose(mean_pwc, mean_reference)

def test_simulate_pwl_spring_mass(srom_simulator_fixture):
    '''
    Tests a PWL surrogate against a manual reference solution generated from  
    test_scripts_data/generate_srom_sim_ref_solution.py
    '''

    pwl_surrogate = srom_simulator_fixture.simulate(10, 'PWL', 1e-12)

    output_pwl = pwl_surrogate.sample(np.array([2]))
    output_ref = np.array([[14.69958116]])

    assert np.isclose(output_pwl, output_ref)

def test_optimization_parameters(srom_simulator_fixture):
    srom_simulator_fixture.set_optimization_parameters(weights=[2,3,5],
                                                       num_tests=750,
                                                       error='MEAN',
                                                       max_moment=7,
                                                       cdf_grid_pts=75,
                                                       tolerance=.01,
                                                       options={'maxiter': 800,
                                                                'disp': False},
                                                       method='CG',
                                                       joint_opt=False)
    
    assert srom_simulator_fixture._weights == [2,3,5]
    assert srom_simulator_fixture._num_test_samples == 750
    assert srom_simulator_fixture._error == 'MEAN'
    assert srom_simulator_fixture._max_moment == 7
    assert srom_simulator_fixture._cdf_grid_pts == 75
    assert srom_simulator_fixture._tolerance == .01
    assert srom_simulator_fixture._options == {'maxiter': 800, 'disp': False}
    assert srom_simulator_fixture._method == 'CG'
    assert srom_simulator_fixture._joint_opt == False 

def test_evaluate_model_for_samples_return_type(srom_simulator_fixture,
                                                srom_base_fixture):
    (samples, _) = srom_base_fixture.get_params()

    output = \
        srom_simulator_fixture.evaluate_model_for_samples(samples)

    assert isinstance(output, np.ndarray)

def test_compute_pwl_gradient_return_type(srom_simulator_fixture,
                                          srom_base_fixture):
    pwl_step_size = 1e-12

    (samples, _) = srom_base_fixture.get_params()

    output = \
        srom_simulator_fixture.evaluate_model_for_samples(samples)

    samples_fd = \
        FD.get_perturbed_samples(samples,
                                 perturbation_values=[pwl_step_size])

    test_gradient = \
        srom_simulator_fixture._compute_pwl_gradient(output,
                                                     samples_fd,
                                                     pwl_step_size)
    assert isinstance(test_gradient, np.ndarray)

def test_simulate_return_type(srom_simulator_fixture):
    test_pwc_surrogate = \
        srom_simulator_fixture.simulate(srom_size=10,
                                        surrogate_type='PWC')

    test_pwl_surrogate = \
        srom_simulator_fixture.simulate(srom_size=10,
                                        surrogate_type='PWL',
                                        pwl_step_size=1e-12)

    assert isinstance(test_pwc_surrogate, SROMSurrogate)
    assert isinstance(test_pwl_surrogate, SROMSurrogate)
