""" Components that have an analytic frequency emission law

These classes have to provide the following functionality:
    - are constructed from an arbitrary analytic expression
    - provide an efficient evaluator that takes as argument frequencies and
      parameters
    - same thing for the gradient wrt the parameters

For frequent components (e.g. power law, gray body) these classe are already
prepared.
"""

import numpy as np
import sympy
from sympy import lambdify, Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.autowrap import ufuncify
from scipy import constants
from astropy.cosmology import Planck15

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
            return param[..., np.newaxis]
        else:
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

        res = np.zeros(shape)
        for i_p, p in enumerate(new_params):
            res[i_p] += self._lambda_diff[i_p](nu, new_params[i_p])
        return list(res)

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
            self.defaults.append(self._REF_BETA)

        if temp is None:
            self.defaults.append(self._REF_TEMP)


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
            self.defaults.append(self._REF_BETA)


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
        
        super(PowerLawWithCurvature, self).__init__(analytic_expr, **kwargs)

        if beta_pl is None:
            self.defaults.append(self._REF_BETA)

        if running is None:
            self.defaults.append(self._REF_RUN)

        if nu_pivot is None:
            self.defaults.append(self._REF_NU_PIVOT)

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
            self.evaluate = lambda nu: np.ones_like(nu)


class Dust(ModifiedBlackBody):
    ''' Alias of ModifiedBlackBody
    '''
    pass


class Synchrotron(PowerLaw):
    ''' Alias of PowerLaw
    '''
    pass
