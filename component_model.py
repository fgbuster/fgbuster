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
        self._n_param = len(self._expr.free_symbols) 
        if self._expr.has(Symbol('nu')):
            self._n_param -= 1
        str_symbols = ['nu'] + ['param_%i'%i for i in range(self._n_param)]
        self._symbols = sympy.symbols(str_symbols)
        self._lambda = lambdify(self._symbols, self._expr, 'numpy')
        lambdify_diff_par_i = lambda i: lambdify(
            self._symbols, self._expr.diff('param_%i'%i), 'numpy')
        self._lambda_diff = [lambdify_diff_par_i(i)
                              for i in range(self._n_param)]
        self._defaults = []

    def _add_last_dimention_if_not_scalar(self, param):
        if isinstance(param, np.ndarray) and len(param) > 1:
            return param[..., np.newaxis]
        else:
            return param

    def eval(self, nu, *params):
        assert len(params) == self.__n_param
        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions: 
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self._add_last_dimention_if_not_scalar, params)
        return self._lambda(nu, *new_params)

    def gradient(self, nu, *params):
        assert len(params) == self._n_param
        if not params:
            return 0.
        elif len(np.broadcast(*params).shape) <= 1:
            # Parameters are all scalars.
            # This case is frequent and easy, thus leave early
            return self._lambda_diff[0](nu, *params)[np.newaxis, ...]

        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions: 
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self._add_last_dimention_if_not_scalar, params)

        shape = (self._n_param,) + np.broadcast(*params).shape + (len(nu),)
        res = np.zeros(shape)
        for i_p, p in enumerate(new_params):
            res[i_p] += self._lambda_diff[i_p](nu, new_params[i_p])
        return res

    @property
    def n_param(self):
        return self._n_param

    @property
    def defaults(self):
        return self._defaults


class ModifiedBlackBody(Component):
    _REF_BETA = 19.6
    _REF_TEMP = 1.6

    def __init__(self, nu0, temp=None, beta=None, units='K_CMB'):
        # Prepare the analytic expression
        # Note: beta_d (not beta) avoids collision with sympy beta functions
        analytic_expr = ('expm1(nu0 / temp * h_over_k)'
                         '/ expm1(nu / temp * h_over_k)'
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
        kwargs = {'nu0': nu0}
        if temp is None:
            analytic_expr = analytic_expr.replace('temp', 'param_0')
            self.ref_param_0 = self.ref_temp
        else:
            kwargs['temp'] = temp
            self.ref_param_1 = self.ref_temp
        if beta is None:
            if temp is None:
                beta_par_tag = 'param_1'
            else:
                beta_par_tag = 'param_0'
                self.ref_param_0 = self.ref_beta
            analytic_expr = analytic_expr.replace('beta_d', beta_par_tag)
        else:
            kwargs['beta_d'] = beta
        kwargs['h_over_k'] = H_OVER_K
        
        super(ModifiedBlackBody, self).__init__(analytic_expr, **kwargs)

        if temp is None:
            self.defaults.append(_REF_TEMP)

        if beta is None:
            self.defaults.append(_REF_BETA)



class PowerLaw(Component):
    _REF_BETA = -3

    def __init__(self, nu0, beta=None, units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = ('(nu / nu0)**(param_0)')
        if units == 'K_CMB':
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif units == 'K_RJ':
            pass
        else:
            raise ValueError('Unsupported units: %s'%units)

        kwargs = {'nu0': nu0}
        if beta is not None:
            kwargs['param_0'] = beta
            self.ref_param_0 = self.ref_beta
        
        super(PowerLaw, self).__init__(analytic_expr, **kwargs)

        if beta is None:
            self.defaults.append(_REF_BETA)


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
