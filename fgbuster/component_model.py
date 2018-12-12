""" Parametric spectral energy distribution (SED)

Unified API for evaluating SEDs, see :class:`Component`.

This module also provides a handy way of generating a :class:`Component` from
analytic expressions, see the :class:`AnalyticComponent`. For components
frequently used (e.g. power law, gray body, CMB) these are already
prepared.
"""

import os.path as op
import numpy as np
import sympy
from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr
import scipy
from scipy import constants
from astropy.cosmology import Planck15
import pysm


__all__ = [
    'Component',
    'AnalyticComponent',
    'AME',
    'CMB',
    'Dust',
    'FreeFree',
    'ModifiedBlackBody',
    'PowerLaw',
    'PowerLawCurv',
    'Synchrotron',
]


H_OVER_K = constants.h * 1e9 / constants.k

# Conversion factor at frequency nu
K_RJ2K_CMB = ('(expm1(h_over_k * nu / Tcmb)**2'
              '/ (exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb)**2))')
K_RJ2K_CMB = K_RJ2K_CMB.replace('Tcmb', str(Planck15.Tcmb(0).value))
K_RJ2K_CMB = K_RJ2K_CMB.replace('h_over_k', str(H_OVER_K))

# Conversion factor at frequency nu divided by the one at frequency nu0
K_RJ2K_CMB_NU0 = K_RJ2K_CMB + ' / ' + K_RJ2K_CMB.replace('nu', 'nu0')


class Component(object):
    """ Abstract class for SED evaluation

    It defines the API.
    """

    def _add_last_dimension_if_not_scalar(self, param):
        if isinstance(param, np.ndarray) and len(param) > 1:
            # Lambdified expressions always output an ndarray with shape
            # (param_dim_1, ..., param_dim_n, n_freq). However, parameters and
            # frequencies are both symbols with (no special meaning). In order
            # to impose the shape of the output, we append a dimension to the
            # parameters and let the broadcasting inside the lambdified
            # expressions do the rest
            # TODO: Can replace this with a try/except?
            return param[..., np.newaxis]
        # param is an scalar value, no special treatment is required
        return param

    def eval(self, nu, *params):
        """ Evaluate the SED

        Parameters
        ----------
        nu: array
            Frequencies at which the SED is evaluated
        *params: float or ndarray
            Value of each of the free parameters. They can be arrays and, in
            this case, they should be broadcastable to a common shape.

        Returns
        -------
        result: ndarray
            SED. The shape is
            * the same of `nu` the parameters are all scalar. 
            * `np.broadcast(*params).shape + nu.shape` otherwise
        """
        assert len(params) == self.n_param
        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions:
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self._add_last_dimension_if_not_scalar, params)
        return self._lambda(nu, *new_params)

    def diff(self, nu, *params):
        """ Evaluate the derivative of the SED

        Parameters
        ----------
        nu: array
            Frequencies at which the SED is evaluated
        *params: float or ndarray
            Value of the free parameters. They can be arrays and, in this case,
            they should be broadcastable to a common shape.

        Returns
        -------
        result: list
            It contains the derivative with respect to each parameter. See
            :meth:`eval` for more details about the format of the
            evaluated derivative
        """
        assert len(params) == self.n_param
        if not params:
            return []
        elif len(np.broadcast(*params).shape) <= 1:
            # Parameters are all scalars.
            # This case is frequent and easy, thus leave early
            return [self._lambda_diff[i_p](nu, *params)
                    for i_p in range(self.n_param)]

        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions:
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self._add_last_dimension_if_not_scalar, params)

        res = []
        for i_p in range(self.n_param):
            res.append(self._lambda_diff[i_p](nu, *new_params))
        return res

    def diff_diff(self, nu, *params):
        assert len(params) == self.n_param
        if not params:
            return []
        elif len(np.broadcast(*params).shape) <= 1:
            # Parameters are all scalars.
            # This case is frequent and easy, thus leave early
            return [[self._lambda_diff_diff[i_p][j_p](nu, *params)
                     for i_p in range(self.n_param)]
                    for j_p in range(self.n_param)]

        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions:
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self._add_last_dimension_if_not_scalar, params)

        res = []
        for i_p in range(self.n_param):
            res.append([self._lambda_diff[i_p][j_p](nu, *new_params)
                        for j_p in range(self.n_param)])
        return res

    @property
    def params(self):
        """ Name of the free parameters
        """
        return self._params

    @property
    def n_param(self):
        """ Number of free parameters
        """
        return len(self._params)

    @property
    def defaults(self):
        """ Default values of the free parameters
        """
        try:
            assert len(self._defaults) == self.n_param
        except (AttributeError, AssertionError):
            print("Component: uninitialized defaults requested, returning ones")
            return [1.] * self.n_param
        return self._defaults

    @defaults.setter
    def defaults(self, new_defaults):
        assert len(new_defaults) == self.n_param, ("The length of the defaults"
                                                   "should be %i"%self.n_param)
        self._defaults = new_defaults

    def __getattr__(self, attr):
        # Helpful messages when virtual attribute are not defined
        message = ("Attempt to either use a bare 'Component' object or to"
                   "use an incomplete child class.")
        if attr == '_lambda':
            message += (" Child classes should store in '_lambda'"
                        "the bare SED evaluator or, alternatively, override"
                        "'Component.eval'")
        elif attr == '_lambda_diff':
            message += (" Child classes should store in '_lambda_diff'"
                        "the list of bare evaluators of the derivative of"
                        "the SED for each parameter or, alternatively,"
                        "override 'Component.diff'")
        elif attr == '_lambda_diff_diff':
            message += (" Child classes should store in '_lambda_diff_diff'"
                        "the list of lists of the bare evaluators of the "
                        "second derivatives of the the SED for each"
                        "combination of parameters or, alternatively,"
                        "override 'Component.diff_diff'")
        elif attr == '_params':
            message += (" Child classes should store in '_params'"
                        "the list of the free parameters")
        else:
            raise AttributeError("'%s' object has no attribute '%s'"
                                 % (type(self).__name__, attr))
        raise NotImplementedError(message)


class AnalyticComponent(Component):
    """ Component defined analytically

    Class that allows analytic definition and automatic (symbolic)
    differentiation of it using `sympy`_.


    Parameters
    ----------
    analytic_expr: str
        Analytic expression for the SED. The variable representing the
        frequency is assumed to be ``nu``. You can not use names that produce
        clashes with `sympy`_ definitions (e.g, `functions`_).
        Notable forbidden names are *beta*, *gamma*.
    **fixed_params: float
        Fix the value of the desired variables. If a variable is not specified
        or is set eaual to ``None``, it will be a free parameters.

    Note
    ----
    Difference with respect to a `sympy.Expression`

    * Efficient evaluators of the SED and its derivatives are prepared at
      construction time
    * Following the API specified in :class:`Component`, ``nu`` has a special
      meaning and has a dedicated dimension (the last one) when evaluations are
      performed
    * ``diff`` (and ``diff_diff``) return the evaluation of the derivatives with
      respect to all the free parameters, not the expression of the
      derivatives with respect to a specific parameter

    Note also that

    * You can trade a longer construction time for faster evaluation time by
      setting ``component_model.lambdify`` to
      ``sympy.utilities.autowrap.ufuncify``.
      After constructing the anlytic component you can revert back the change by
      setting ``component_model.lambdify`` back to ``sympy.lambdify``.

    .. _functions: https://docs.sympy.org/latest/modules/functions/index.html
    .. _sympy: https://docs.sympy.org/latest/modules/functions/index.html
    """

    def __init__(self, analytic_expr, **fixed_params):
        self._fixed_params = fixed_params
        self._expr = parse_expr(analytic_expr).subs(fixed_params)
        self._params = sorted([str(s) for s in self._expr.free_symbols])
        self._defaults = []

        # NOTE: nu is in symbols (at index 0) but it is not in self._params
        if 'nu' in self._params:
            self._params.pop(self._params.index('nu'))
        self._params.insert(0, 'nu')
        symbols = sympy.symbols(self._params)
        self._params.pop(0)

        # Create lambda functions
        self._lambda = lambdify(symbols, self._expr, 'numpy')
        lambdify_diff_param = lambda param: lambdify(
            symbols, self._expr.diff(param), 'numpy')
        self._lambda_diff = [lambdify_diff_param(p) for p in self._params]
        lambdify_diff_diff_params = lambda param1, param2: lambdify(
            symbols, self._expr.diff(param1, param2), 'numpy')
        self._lambda_diff_diff = []
        for p1 in self._params:
            self._lambda_diff_diff.append(
                [lambdify_diff_diff_params(p1, p2) for p2 in self._params])

    def __repr__(self):
        return repr(self._expr)


class ModifiedBlackBody(AnalyticComponent):
    """ Modified Black body

    Parameters
    ----------
    nu0: float
        Reference frequency
    temp:
        Black body temperature
    beta_d: float
        Spectral index
    units:
        Output units (K_CMB and K_RJ available)
    """
    _REF_BETA = 1.54
    _REF_TEMP = 20.

    def __init__(self, nu0, temp=None, beta_d=None, units='K_CMB'):
        # Prepare the analytic expression
        # Note: beta_d (not beta) avoids collision with sympy beta functions
        #TODO: Use expm1 and get Sympy processing it as a symbol
        analytic_expr = ('(exp(nu0 / temp * h_over_k) -1)'
                         '/ (exp(nu / temp * h_over_k) - 1)'
                         '* (nu / nu0)**(1 + beta_d)')
        if 'K_CMB' in units:
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        # Parameters in the analytic expression are
        # - Fixed parameters -> into kwargs
        # - Free parameters -> renamed according to the param_* convention
        kwargs = {
            'nu0': nu0, 'beta_d': beta_d, 'temp': temp, 'h_over_k': H_OVER_K
        }

        super(ModifiedBlackBody, self).__init__(analytic_expr, **kwargs)

        if beta_d is None:
            self._defaults.append(self._REF_BETA)

        if temp is None:
            self._defaults.append(self._REF_TEMP)


class PowerLaw(AnalyticComponent):
    """ Power law

    Parameters
    ----------
    nu0: float
        Reference frequency
    beta_pl: float
        Spectral index
    units:
        Output units (K_CMB and K_RJ available)
    """
    _REF_BETA = -3

    def __init__(self, nu0, beta_pl=None, units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = ('(nu / nu0)**(beta_pl)')
        if 'K_CMB' in units:
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        kwargs = {'nu0': nu0, 'beta_pl': beta_pl}

        super(PowerLaw, self).__init__(analytic_expr, **kwargs)

        if beta_pl is None:
            self._defaults.append(self._REF_BETA)


class PowerLawCurv(AnalyticComponent):
    """ Power law with curvature

    Parameters
    ----------
    nu0: float
        Reference frequency
    nu_pivot: float
        Pivot frequency for the running
    beta_pl: float
        Spectral index
    running: float
        Curvature of the Synchrotron
    units:
        Output units (K_CMB and K_RJ available)
    """
    _REF_BETA = -3.
    _REF_RUN = 0.

    def __init__(self, nu0, nu_pivot, beta_pl=None, running=None,
                 units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = '(nu / nu0)**(beta_pl + running * log(nu / nu_pivot))'
        if 'K_CMB' in units:
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        kwargs = {'nu0': nu0, 'nu_pivot': nu_pivot,
                  'beta_pl': beta_pl, 'running': running}

        super(PowerLawCurv, self).__init__(analytic_expr, **kwargs)

        if beta_pl is None:
            self._defaults.append(self._REF_BETA)

        if running is None:
            self._defaults.append(self._REF_RUN)


class CMB(AnalyticComponent):
    """ Cosmic microwave background

    Parameters
    ----------
    units:
        Output units (K_CMB and K_RJ available)
    """

    def __init__(self, units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = ('1')
        if units == 'K_CMB':
            pass
        elif units == 'K_RJ':
            analytic_expr += ' / ' + K_RJ2K_CMB
        else:
            raise ValueError("Unsupported units: %s"%units)

        super(CMB, self).__init__(analytic_expr)

        if 'K_CMB' in units:
            self.eval = lambda nu: np.ones_like(nu)


class FreeFree(AnalyticComponent):
    """ Free-free
    
    Anlytic model for bremsstrahlung emission (Draine, 2011)
    Above 1GHz it is essentially equivalent to a power law.

    Parameters
    ----------
    logEM:
        Logarithm (base ten) of the integrated squared electron density along a
        line of sight in cm^-3 pc
    Te:
        Electron temperature
    units:
        Output units (K_CMB and K_RJ available)
    """

    _REF_LOGEM = 0.
    _REF_TE = 7000.

    def __init__(self, logEM=None, Te=None, units='K_CMB'):
        # Prepare the analytic expression. Planck15 X, Table 4
        # NOTE: PySM uses power a power law instead
        T4 = 'Te * 1e-4'
        gff = 'log(exp(5.960 - (sqrt(3) / pi) * log(nu * (T4)**(-3 / 2))) + exp(1))'
        tau = '0.05468 * Te**(- 3 / 2) / nu**2 * 10**(EM) * (gff)'
        analytic_expr = '1e6 * Te * (1 - exp(-(tau)))'
        analytic_expr = analytic_expr.replace('tau', tau)
        analytic_expr = analytic_expr.replace('gff', gff)
        analytic_expr = analytic_expr.replace('T4', T4)
        if 'K_CMB' in units:
            analytic_expr += ' * ' + K_RJ2K_CMB
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        kwargs = dict(logEM=logEM, Te=Te, tau=tau, gff=gff, T4=T4)

        super(FreeFree, self).__init__(analytic_expr, **kwargs)

        if logEM is None:
            self._defaults.append(self._REF_LOGEM)

        if Te is None:
            self._defaults.append(self._REF_TE)


class AME(Component):
    _REF_NU_PEAK = 30.
    _NU_PEAK_0 = 30.

    def __init__(self, nu_0, nu_peak=None, units='K_CMB'):
        # analytic_expr contains just the analytic part of the emission law
        analytic_expr = 'nu**(-2)'
        if 'K_CMB' in units:
            analytic_expr += '*' + K_RJ2K_CMB
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        super(AME, self).__init__(analytic_expr)

        emissivity_file = op.join(op.dirname(pysm.__file__),
                                  'template/emissivity.txt')
        emissivity = np.loadtxt(emissivity_file, unpack=True)
        self._interp = scipy.interpolate.interp1d(
            emissivity[0], emissivity[1],
            bounds_error=False, fill_value=0, assume_sorted=True, copy=False)

        if nu_peak is None:
            self._defaults.append(self._REF_NU_PEAK)
            self._params.append('nu_peak')
            self.eval = lambda nu, p: (
                self._interp_eval(nu, p) / self._interp_eval(nu_0, p))
            self.diff = lambda nu, p: [
                (self.eval(nu, p*1.01) - self.eval(nu, p*0.99)) / (p*0.02)]
        else:
            self.eval = lambda nu: (self._interp_eval(nu, nu_peak)
                                    / self._interp_eval(nu_0, nu_peak))
            self.diff = lambda nu: []

    def _interp_eval(self, nu, nu_peak):
        return self._lambda(nu) * self._interp(nu * (self._NU_PEAK_0 / nu_peak))


class Dust(ModifiedBlackBody):
    """ Alias of :class:`ModifiedBlackBody`
    """
    pass


class Synchrotron(PowerLaw):
    """ Alias of :class:`PowerLaw`
    """
    pass
