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
        self.__analytic_expr = analytic_expr
        self.__fixed_params = fixed_params
        self.__expr = parse_expr(analytic_expr).subs(fixed_params)
        self.__n_param = len(self.__expr.free_symbols) 
        if self.__expr.has(Symbol('nu')):
            self.__n_param -= 1
        str_symbols = ['nu'] + ['param_%i'%i for i in range(self.__n_param)]
        self.__symbols = sympy.symbols(str_symbols)
        self.__lambda = lambdify(self.__symbols, self.__expr, 'numpy')
        lambdify_diff_par_i = lambda i: lambdify(
            self.__symbols, self.__expr.diff('param_%i'%i), 'numpy')
        self.__lambda_diff = [lambdify_diff_par_i(i)
                              for i in range(self.__n_param)]

    def __add_last_dimention_if_not_scalar(self, param):
        if isinstance(param, np.ndarray) and len(param) > 1:
            return param[..., np.newaxis]
        else:
            return param

    def eval(self, nu, *params):
        assert len(params) == self.__n_param
        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions: 
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self.__add_last_dimention_if_not_scalar, params)
        return self.__lambda(nu, *new_params)

    def gradient(self, nu, *params):
        assert len(params) == self.__n_param
        if not params:
            return 0.
        elif len(np.broadcast(*params).shape) <= 1:
            # Parameters are all scalars.
            # This case is frequent and easy, thus leave early
            return self.__lambda_diff[0](nu, *params)[np.newaxis, ...]

        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions: 
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = map(self.__add_last_dimention_if_not_scalar, params)

        shape = (self.__n_param,) + np.broadcast(*params).shape + (len(nu),)
        res = np.zeros(shape)
        for i_p, p in enumerate(new_params):
            res[i_p] += self.__lambda_diff[i_p](nu, new_params[i_p])
        return res

    @property
    def n_param(self):
        return self.__n_param


class ModifiedBlackBody(Component):

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
        else:
            kwargs['temp'] = temp
        if beta is None:
            beta_par_tag = 'param_1' if temp is None else 'param_0'
            analytic_expr = analytic_expr.replace('beta_d', beta_par_tag)
        else:
            kwargs['beta_d'] = beta
        kwargs['h_over_k'] = H_OVER_K
        
        super(ModifiedBlackBody, self).__init__(analytic_expr, **kwargs)


class PowerLaw(Component):

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
        
        super(PowerLaw, self).__init__(analytic_expr, **kwargs)


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
