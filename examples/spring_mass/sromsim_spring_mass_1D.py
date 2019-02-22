"""
This is a simple example meant to demonstrate SROMPy's functionality with the
SROMSimulator class. Estimates the maximum displacement of a spring-mass system
with a random stiffness using SROMs and compares the solution to Monte Carlo
simulation.
"""

import numpy as np

from spring_mass_model import SpringMassModel
from SROMPy.srom import SROMSimulator
from SROMPy.postprocess import Postprocessor
from SROMPy.target import BetaRandomVariable, SampleRandomVector

# Random variable for spring stiffness
stiffness_random_variable = \
    BetaRandomVariable(alpha=3., beta=2., shift=1., scale=2.5)

# Specify spring-mass system:
m = 1.5                             # Deterministic mass.
state0 = [0., 0.]                   # Initial conditions.
time_step = 0.01

# Initialize model:
model = SpringMassModel(m, state0=state0, time_step=time_step)

# ----------Monte Carlo------------------
print "Generating Monte Carlo reference solution..."

# Generate stiffness input samples for Monte Carlo:
num_samples = 5000
stiffness_samples = stiffness_random_variable.draw_random_sample(num_samples)

# Calculate maximum displacement samples using MC simulation:
displacement_samples = np.zeros(num_samples)
for i, stiff in enumerate(stiffness_samples):
    displacement_samples[i] = model.evaluate([stiff])

# Get Monte carlo solution as a sample-based random variable:
monte_carlo_solution = SampleRandomVector(displacement_samples)

# ----------SROMSimulator----------------
print "Generating SROMSimulator for input (stiffness)..."

srom_sim = SROMSimulator(stiffness_random_variable, model)

print "Computing Piecewise Constant SROMSurrogate approximation for output..."
srom_size = 10

pwc_srom_surrogate = srom_sim.simulate(srom_size, "PWC")

# Compare solutions:
pp_pwc_output = Postprocessor(pwc_srom_surrogate, monte_carlo_solution)
pp_pwc_output.compare_cdfs(pwc_srom_surrogate, monte_carlo_solution)

# Compare mean estimates for output:
print "Monte Carlo mean estimate: ", np.mean(displacement_samples)
print "SROM mean estimate: ", pwc_srom_surrogate.compute_moments(1)[0][0]

# ----------Piecewise Linear Function----------------
print "Computing Piecewise Linear SROMSurrogate approximation for output..."

pwl_srom_surrogate = srom_sim.simulate(srom_size, "PWL", 1e-12)

# Compare solutions:
pp_pwl_output = Postprocessor(pwl_srom_surrogate, monte_carlo_solution)
pp_pwl_output.compare_cdfs(pwl_srom_surrogate, monte_carlo_solution)
