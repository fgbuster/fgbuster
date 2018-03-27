""" Components that have an analytic frequency emission law

These classes have to provide the following functionality:
    - are constructed from an arbitrary analytic expression
    - provide an efficient evaluator that takes as argument frequencies and
      parameters
    - same thing for the gradient wrt the parameters

For frequent components (e.g. power law, gray body) these classe are already
prepared.
"""

import os.path as op
import numpy as np
import sympy
from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.autowrap import ufuncify
import scipy
from scipy import constants
from astropy.cosmology import Planck15
import pysm

H_OVER_K = constants.h * 1e9 / constants.k

# Conversion factor at frequency nu
K_RJ2K_CMB = ('(expm1(h_over_k * nu / Tcmb)**2'
              '/ (exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb)**2))')
K_RJ2K_CMB = K_RJ2K_CMB.replace('Tcmb', str(Planck15.Tcmb(0).value))
K_RJ2K_CMB = K_RJ2K_CMB.replace('h_over_k', str(H_OVER_K))

# Conversion factor at frequency nu divided by the one at frequency nu0
K_RJ2K_CMB_NU0 = K_RJ2K_CMB + ' / ' + K_RJ2K_CMB.replace('nu', 'nu0')


class Component(object):

    def __init__(self, analytic_expr, **fixed_params):
        self._analytic_expr = analytic_expr
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

    def _add_last_dimention_if_not_scalar(self, param):
        if isinstance(param, np.ndarray) and len(param) > 1:
            # Lambdified expressions always output an ndarray with shape
            # (param_dim_1, ..., param_dim_n, n_freq). However, parameters and
            # frequencies are both symbols with (no special meaning). In order
            # to impose the shape of the output we append a dimesion to the
            # parameters and let the broadcasting inside the lambdified
            # expressions do the rest
            return param[..., np.newaxis]
        # param is an scalar value, no special treatment is required
        return param

    def eval(self, nu, *params):
        assert len(params) == self.n_param
        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions:
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self._add_last_dimention_if_not_scalar, params)
        return self._lambda(nu, *new_params)

    def gradient(self, nu, *params):
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
        new_params = map(self._add_last_dimention_if_not_scalar, params)

        res = []
        for i_p, p in enumerate(new_params):
            res.append(self._lambda_diff[i_p](nu, p))
        return res

    @property
    def params(self):
        ''' Name of the parameters (alphabetical order)
        '''
        return self._params

    @property
    def n_param(self):
        return len(self._params)

    @property
    def defaults(self):
        return self._defaults

    @defaults.setter
    def defaults(self, new_defaults):
        assert len(self._defaults) == len(new_defaults)
        self._defaults = new_defaults


class ModifiedBlackBody(Component):
    _REF_BETA = 1.54
    _REF_TEMP = 20.

    def __init__(self, nu0, temp=None, beta_d=None, units='K_CMB'):
        # Prepare the analytic expression
        # Note: beta_d (not beta) avoids collision with sympy beta functions
        #TODO: Use expm1 and get Sympy processing it as a symbol
        analytic_expr = ('(exp(nu0 / temp * h_over_k) -1)'
                         '/ (exp(nu / temp * h_over_k) - 1)'
                         '* (nu / nu0)**(1 + beta_d)')
        if units == 'K_CMB':
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif units == 'K_RJ':
            pass
        else:
            raise ValueError('Unsupported units: %s'%units)

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


class PowerLaw(Component):
    _REF_BETA = -3

    def __init__(self, nu0, beta_pl=None, units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = ('(nu / nu0)**(beta_pl)')
        if units == 'K_CMB':
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif units == 'K_RJ':
            pass
        else:
            raise ValueError('Unsupported units: %s'%units)

        kwargs = {'nu0': nu0, 'beta_pl': beta_pl}

        super(PowerLaw, self).__init__(analytic_expr, **kwargs)

        if beta_pl is None:
            self._defaults.append(self._REF_BETA)


class PowerLawCurv(Component):
    _REF_BETA = -3.
    _REF_RUN = 0.
    _REF_NU_PIVOT = 70.

    def __init__(self, nu0, beta_pl=None, running=None, nu_pivot=None, units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = ('(nu / nu0)**(beta_pl + running * log( nu / nu_pivot ))')
        if units == 'K_CMB':
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif units == 'K_RJ':
            pass
        else:
            raise ValueError('Unsupported units: %s'%units)

        kwargs = {'nu0': nu0, 'beta_pl': beta_pl, 'running': running, 'nu_pivot': nu_pivot}

        super(PowerLawCurv, self).__init__(analytic_expr, **kwargs)

        if beta_pl is None:
            self._defaults.append(self._REF_BETA)

        if running is None:
            self._defaults.append(self._REF_RUN)

        if nu_pivot is None:
            self._defaults.append(self._REF_NU_PIVOT)


class CMB(Component):

    def __init__(self, units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = ('1')
        if units == 'K_CMB':
            pass
        elif units == 'K_RJ':
            analytic_expr += ' / ' + K_RJ2K_CMB
        else:
            raise ValueError('Unsupported units: %s'%units)

        super(CMB, self).__init__(analytic_expr)

        if units == 'K_CMB':
            self.eval = lambda nu: np.ones_like(nu)


class FreeFree(Component):
    _REF_EM = 15.
    _REF_TE = 7000.

    def __init__(self, EM=None, Te=None, units='K_CMB'):
        # Prepare the analytic expression. Planck15 X, Table 4
        # NOTE: PySM uses power a power law instead
        T4 = 'Te * 1e-4'
        gff = 'log(exp(5.960 - (sqrt(3) / pi) * log(nu * (T4)**(-3 / 2))) + exp(1))'
        tau = '0.05468 * Te**(- 3 / 2) / nu**2 * EM * (gff)'
        analytic_expr = '1e6 * Te * (1 - exp(-(tau)))'
        analytic_expr = analytic_expr.replace('tau', tau)
        analytic_expr = analytic_expr.replace('gff', gff)
        analytic_expr = analytic_expr.replace('T4', T4)
        if units == 'K_CMB':
            analytic_expr += ' * ' + K_RJ2K_CMB
        elif units == 'K_RJ':
            pass
        else:
            raise ValueError('Unsupported units: %s'%units)

        kwargs = dict(EM=EM, Te=Te, tau=tau, gff=gff, T4=T4)

        super(FreeFree, self).__init__(analytic_expr, **kwargs)

        if EM is None:
            self._defaults.append(self._REF_EM)

        if Te is None:
            self._defaults.append(self._REF_TE)


class AME(Component):
    _REF_NU_PEAK = 30.
    _NU_PEAK_0 = 30.

    def __init__(self, nu_0, nu_peak=None, units='K_CMB'):
        # analytic_expr contains just the analytic part of the emission law
        analytic_expr = 'nu**(-2)'
        if units == 'K_CMB':
            analytic_expr += '*' + K_RJ2K_CMB
        elif units == 'K_RJ':
            pass
        else:
            raise ValueError('Unsupported units: %s'%units)

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
            self.gradient = lambda nu, p: [
                (self.eval(nu, p*1.01) - self.eval(nu, p*0.99)) / (p*0.02)]
        else:
            self.eval = lambda nu: (self._interp_eval(nu, nu_peak)
                                    / self._interp_eval(nu_0, nu_peak))
            self.gradient = lambda nu: []

    def _interp_eval(self, nu, nu_peak):
        return self._lambda(nu) * self._interp(nu * (self._NU_PEAK_0 / nu_peak))


class Dust(ModifiedBlackBody):
    ''' Alias of ModifiedBlackBody
    '''
    pass


class Synchrotron(PowerLaw):
    ''' Alias of PowerLaw
    '''
    pass
