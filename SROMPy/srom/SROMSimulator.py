import numpy as np

from SROMPy.model import Model
from SROMPy.target.RandomVariable import RandomVariable
from SROMPy.srom import SROM, FiniteDifference as FD, SROMSurrogate

class SROMSimulator(object):
    """
    Computes an estimate based on the Stochastic Reduced Order Model
    algorithm.
    """
    def __init__(self, random_input, model):
        """
        Requires a data object that provides input samples and a model.

        :param random_input: Provides a data sampling function.
        :type random_input: Input
        :param model: A model that outputs from a sample data input.
        :type model: Model
        :param enhanced_optimize: A model that outputs from a sample data input.
        :type enhanced_optimize: bool
        """
        self.__check_init_parameters(random_input, model)

        self._random_variable_data = random_input
        self._model = model
        self._dim = random_input.dim
        self._weights = [1, 1, 1]
        self._num_test_samples = 50
        self._error = 'SSE'
        self._max_moment = 5
        self._cdf_grid_pts = 100
        self._tolerance = None
        self._options = None
        self._method = None
        self._joint_opt = False

    def simulate(self, srom_size, surrogate_type, pwl_step_size=None):
        """
        Performs the SROM Simulation.
        Can specify which function to use: Piecewise Constant or Linear.
        If Piecewise Linear is the chosen, must include a step size.

        :param srom_size: Size of SROM.
        :type srom_size: int
        :param surrogate_type: The SROM type being simulated. Piecewise constant
            and piecewise linear are currently implemented.
        :type surrogate_type: str
        :param pwl_step_size: Step size for piecewise linear, defaults to None.
        :param pwl_step_size: float, optional
        :return: Returns a SROM surrogate object.
        :rtype: SROMSurrogate
        """
        self.__check_simulate_parameters(srom_size,
                                         surrogate_type,
                                         pwl_step_size)

        input_srom = self._generate_input_srom(srom_size)
        (samples, _) = input_srom.get_params()

        if surrogate_type == "PWC":
            output_samples = \
                self._simulate_piecewise_constant(samples)

            output_gradients = None

        elif surrogate_type == "PWL":
            output_samples, output_gradients = \
                self._simulate_piecewise_linear(samples, pwl_step_size)

        srom_surrogate = \
            SROMSurrogate(input_srom, output_samples, output_gradients)

        return srom_surrogate

    def set_optimization_parameters(self,
                                    weights=None,
                                    num_tests=50,
                                    error='SSE',
                                    max_moment=5,
                                    cdf_grid_pts=100,
                                    tolerance=None,
                                    options=None,
                                    method=None,
                                    joint_opt=False):
        """
        Sets the additional optimization parameters for SROM's optimize method.

        :param weights: relative weights specifying importance of matching
            CDFs, moments, and correlation of the target during optimization.
            Default is equal weights [1,1,1].
        :type weights: 1d Numpy array (length = 3)
        :param num_test_samples: Number of sample sets (iterations) to run
            optimization.
        :type num_test_samples: int
        :param error: Type of error metric to use in objective ("SSE", "MAX",
            "MEAN").
        :type error: string
        :param max_moment: Max number of target moments to consider matching
        :type max_moment: int
        :param cdf_grid_pts: Number of points to evaluate CDF error on
        :type cdf_grid_pts: int
        :param tolerance: Tolerance for scipy optimization algorithm.
        :type tolerance: float
        :param options: Scipy optimization algorithm options, see scipy
            documentation.
        :type options: dict
        :param method: Method used for scipy optimization, see scipy
            documentation.
        :type method: string
        :param joint_opt: Flag to optimize jointly for samples & probabilities.
        :type joint_opt: bool
        """
        self._weights = weights
        self._num_test_samples = num_tests
        self._error = error
        self._max_moment = max_moment
        self._cdf_grid_pts = cdf_grid_pts
        self._tolerance = tolerance
        self._options = options
        self._method = method
        self._joint_opt = joint_opt

    def _simulate_piecewise_constant(self, input_samples):
        """
        Performs the simulation of the piecewise constant function.

        :param input_samples: Samples initialized by SROM get_params method.
        :type input_samples: np.ndarray
        :return: Returns the SROM output samples.
        :rtype: np.ndarray
        """
        output_samples = \
            self.evaluate_model_for_samples(input_samples)

        return output_samples

    def _simulate_piecewise_linear(self, input_samples, pwl_step_size):
        """
        Performs the simulation of the piecewise linear function.

        :param input_samples: Samples initialized by SROM get_params method.
        :type input_samples: np.ndarray
        :param pwl_step_size: Step size used to generate the gradient and the
            perturbed samples.
        :type pwl_step_size: float
        :return: Returns the output samples and gradient for SROMSurrogate.
        :rtype: np.ndarray, np.ndarray
        """
        output_samples = \
            self.evaluate_model_for_samples(input_samples)

        samples_fd = \
            FD.get_perturbed_samples(samples=input_samples,
                                     perturbation_values=[pwl_step_size])

        gradient = \
            self._compute_pwl_gradient(output_samples,
                                       samples_fd,
                                       pwl_step_size)

        return output_samples, gradient

    def _compute_pwl_gradient(self, output_samples, samples_fd, step_size):
        """
        Computes the gradient for the piecewise linear function.

        :param output_samples: Samples generated in _simulate_piecewise_linear
            method.
        :type output_samples: np.ndarray
        :param samples_fd: Perturbed samples.
        :type samples_fd: np.ndarray
        :param step_size: Step sized used to compute the gradient.
        :type step_size: float
        :return: Returns the gradient for SROMSurrogate.
        :rtype: np.ndarray
        """
        perturbed_output = \
            self.evaluate_model_for_samples(samples_fd)

        gradient = FD.compute_gradient(output_samples,
                                       perturbed_output,
                                       [step_size])

        return gradient

    def evaluate_model_for_samples(self, input_samples):
        """
        Uses model's evaluate method to return output samples.

        :param input_samples: Samples initialized by SROM get_params method.
        :type input_samples: np.ndarray
        :return: Returns output samples generated by model's evaluate method.
        :rtype: np.ndarray
        """
        output = np.zeros(len(input_samples))

        for i, values in enumerate(input_samples):
            output[i] = self._model.evaluate([values])

        return output

    def _generate_input_srom(self, srom_size):
        """
        Generates an SROM object. If _enhanced_optimize is
        activated, will optimize with given parameters.

        :param srom_size: The size of SROM.
        :type srom_size: int
        :return: Returns an SROM object.
        :rtype: SROM
        """
        srom = SROM(srom_size, self._dim)

        srom.optimize(target_random_variable=self._random_variable_data,
                      weights=self._weights,
                      num_test_samples=self._num_test_samples,
                      error=self._error,
                      max_moment=self._max_moment,
                      cdf_grid_pts=self._cdf_grid_pts,
                      tolerance=self._tolerance,
                      options=self._options,
                      method=self._method,
                      joint_opt=self._joint_opt)

        return srom

    @staticmethod
    def __check_init_parameters(data, model):
        """
        Inspects parameters given to the init method.

        :param data: Input object provided to init().
        :param model: Model object provided to init().
        """
        if not isinstance(data, RandomVariable):
            raise TypeError("Data must inherit from the RandomVariable class")

        if not isinstance(model, Model):
            raise TypeError("Model must inherit from Model class")

    @staticmethod
    def __check_simulate_parameters(srom_size, surrogate_type,
                                    pwl_step_size):
        """
        Inspects the parameters given to the simulate method.

        :param srom_size: srom_size input provided to simulate().
        :param surrogate_type: surrogate_type provided to simulate().
        :param pwl_step_size: pwl_step_size provided to simulate().
        """
        if not isinstance(srom_size, int):
            raise TypeError("SROM size must be an integer")

        if surrogate_type != "PWC" and surrogate_type != "PWL":
            raise ValueError("Surrogate type must be 'PWC' or 'PWL'")
        #Should this be a TypeError or ValueError? Leaning on value(TODO)
        if surrogate_type == "PWL" and pwl_step_size is None:
            raise TypeError("Step size must be initialized for 'PWL' ex: 1e-12")
