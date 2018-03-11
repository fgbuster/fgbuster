""" Recurrent algebraic functions in component separation

"""
import numpy as np
import scipy as sp
import numdifftools as nd
import inspect
from time import time

OPTIMIZE = False


def _inv(m):
    result = np.array(map(np.linalg.inv, m.reshape((-1,)+m.shape[-2:])))
    return result.reshape(m.shape)


def _solve(a, b):
    u, e, v = np.linalg.svd(a, full_matrices=False)
    utb = _mtv(u, b)
    return _mtv(v, utb / e)


def _mv(m, v):
    return np.einsum('...ij,...j->...i', m, v, optimize=OPTIMIZE)


def _mtv(m, v):
    return np.einsum('...ji,...j->...i', m, v, optimize=OPTIMIZE)


def _mm(m, n):
    return np.einsum('...ij,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mtm(m, n):
    return np.einsum('...ji,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mmv(m, w, v):
    return np.einsum('...ij,...jk,...k->...i', m, w, v, optimize=OPTIMIZE)


def _mtmv(m, w, v):
    return np.einsum('...ji,...jk,...k->...i', m, w, v, optimize=OPTIMIZE)


def _mmm(m, w, n):
    return np.einsum('...ij,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)


def _mtmm(m, w, n):
    return np.einsum('...ji,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)


def _T(x):
    # Indexes < -2 are assumed to count diagonal blocks. Therefore the transpose
    # has to swap the last two axis, not reverse the order of all the axis
    try:
        return np.swapaxes(x, -1, -2)
    except ValueError:
        return x

def logL(A, d, invN=None):
    if invN is None:
        try:
            u, _, _ = np.linalg.svd(A, full_matrices=False)
        except np.linalg.linalg.LinAlgError:
            print 'SVD of A failed -> logL = -inf'
            return - np.inf
        return 0.5 * np.linalg.norm(_mtv(u, d))**2
    ANd = _mtmv(A, invN, d)
    return 0.5 * np.sum(ANd * _mv(_inv(_mtmm(A, invN, A)), ANd))


def invAtNA(A, invN=None):
    if invN is None:
        return _inv(_mtm(A, A))
    return _inv(_mtmm(A, invN, A))


def Wd(A, d, invN=None):
    if invN is None:
        Ad = _mtv(A, d)
    else:
        Ad = _mtmv(A, invN, d)
    return _mv(invAtNA(A, invN), Ad)


def W(A, invN=None):
    invAA = invAtNA(A, invN)
    if invN is None:
        return _mm(invAA, _T(A))
    else:
        return _mmm(invAA, _T(A), invN)


def W_dB(A, A_dB, comp_of_dB, invN=None):
    raise NotImplementedError


def W_dB_dB(A, invN=None):
    raise NotImplementedError


def logL_dB(A, d, A_dB, invN=None, comp_of_dB=np.s_[:]):
    """ Derivative of the log likelihood

    Parameters
    ----------
    A : ndarray
        Mixing matrix. Shape `(..., n_freq, n_comp)`
    d: ndarray
        The data vector. Shape `(..., n_freq)`.
    invN: ndarray or None
        The inverse noise matrix. Shape `(..., n_freq, n_freq)`.
    A_dB : ndarray or list of ndarray
        The derivative of the mixing matrix. If list, each entry is the
        derivative with respect to a different parameter.
    comp_of_dB: index or list of indices
        It allows to provide in `A_dB` only the non-zero columns `A`.
        `A_dB` is assumed to be the derivative of `A[..., comp_of_dB]`.
        If a list is provided, also `A_dB` has to be a list and
        `A_dB[i]` is assumed to be the derivative of `A[..., comp_of_dB[i]]`.

    Returns
    -------
    diff : array
        Derivative of the spectral likelihood. If `A_dB` is a list, `diff[i]`
        is computed from `A_dB[i]`.

    Note
    ----
    The `...` in the shape of the arguments denote any extra set of dimentions.
    They have to be compatible among different arguments in the `numpy`
    broadcasting sense.
    """
    if not isinstance(A_dB, list):
        A_dB = [A_dB]

    if isinstance(comp_of_dB, list):
        assert len(A_dB) == len(comp_of_dB)
    else:
        comp_of_dB = [comp_of_dB] * len(A_dB)

    s = Wd(A, d, invN)
    Ds = d - _mv(A, s)
    if invN is not None:
        Ds = _mv(invN, Ds)

    n_param = len(A_dB)
    diff = np.empty(n_param)
    for i in xrange(n_param):
        diff[i] = np.sum(_mv(A_dB[i], s[..., comp_of_dB[i]]) * Ds)

    return diff


def fisher_logL_dB_dB(A, s, A_dB, comp_of_dB, invN=None):
    n_param = len(A_dB)
    fisher = np.empty((n_param, n_param))
    if invN is None:
        u, _, _ = np.linalg.svd(A, full_matrices=False)
        x = [_mtv(u, _mv(A_dB_i, s[..., comp_of_dB[i]]))
             for i, A_dB_i in enumerate(A_dB)]
        for i in xrange(n_param):
            for j in xrange(n_param):
                fisher[i, j] = np.sum(x[i] * x[j])
    else:
        raise NotImplementedError
    return fisher


def _build_bound_inv_logL_and_logL_dB(A_ev, d, A_dB_ev, comp_of_dB):
    """ Produce the functions -logL(x) and -logL_dB(x)

    Keep in the memory the last SVD of A. If x of the next call coincide with
    the last one, recycle the SVD.
    """
    x_old = [None]
    u_old = [None]
    inv_e_old = [None]
    v_old = [None]

    def _update_x_u_e_v(x):
        # If x is different from the last one, update the SVD
        if not np.all(x == x_old[0]):
            A = A_ev(x)
            u_old[0], e_old, v_old[0] = np.linalg.svd(A, full_matrices=False)
            inv_e_old[0] = 1. / e_old
            x_old[0] = x

    def _inv_logL(x):
        _update_x_u_e_v(x)
        return - 0.5 * np.linalg.norm(_mtv(u_old[0], d))**2

    def _inv_logL_dB(x):
        A_dB = A_dB_ev(x)
        diff = np.empty(len(A_dB))

        _update_x_u_e_v(x)
        utd = _mtv(u_old[0], d)
        s = _mtv(v_old[0], utd * inv_e_old[0])
        Dd = d - _mv(u_old[0], utd)
        for i in xrange(len(diff)):
            diff[i] = - np.sum(_mv(A_dB[i], s[..., comp_of_dB[i]]) * Dd)
        return diff

    return _inv_logL, _inv_logL_dB


def comp_sep(A_ev, d, invN, A_dB_ev, comp_of_dB, *minimize_args, **minimize_kargs):
    """ Perform component separation

    Build the (inverse) spectral likelihood and minimize it to estimate the
    parameters of the mixing matrix. Separate the components using the best-fit
    mixing matrix.

    Parameters
    ----------
    A_ev : function
        The evaluator of the mixing matrix. It takes a float or an array as
        argument and returns the mixing matrix, a ndarray with shape
        `(..., n_freq, n_comp)`
    d: ndarray
        The data vector. Shape `(..., n_freq)`.
    invN: ndarray or None
        The inverse noise matrix. Shape `(..., n_freq, n_freq)`.
    A_dB_ev : function
        The evaluator of the derivative of the mixing matrix.
        It returns a list, each entry is the derivative with respect to a
        different parameter.
    comp_of_dB: list of indices
        It allows to provide as output of `A_dB_ev` only the non-zero columns
        `A`. `A_dB_ev(x)[i]` is assumed to be the derivative of
        `A[..., comp_of_dB[i]]`.
    minimize_args: list
        Positional arguments to be passed to `scipy.optimize.minimize`.
        At this moment it just contains `x0`, the initial guess for the spectral
        parameters
    minimize_kargs: dict
        Keyword arguments to be passed to `scipy.optimize.minimize`.
        Notably, it contains the minimization `method`.

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        Result of the spectral likelihood maximisation
	It is the output of `scipy.optimize.minimize`, and thus includes
	- x : (array)
	    Maximum likelihood spectral parameter,
        with the addition of some extra information
	- s : (ndarray)
	    Separated components. Shape `(..., n_comp)`
	- invAtNA : (ndarray)
	    Covariance of the separated components.
            Shape `(..., n_comp, n_comp)`

    Note
    ----
    The `...` in the arguments denote any extra set of dimention. They have to
    be compatible among different arguments in the `numpy` broadcasting sense.
    """
    # Checks input
    if comp_of_dB is not None: # XXX put this in every routine that uses comp_of_dB?
        comp_of_dB = [np.array([c]) for c in comp_of_dB if isinstance(c, (int, long))]
    disp = 'options' in minimize_kargs and 'disp' in minimize_kargs['options']

    # Prepare functions for minmize
    jac = None
    if A_dB_ev is not None and invN is None:
        fun, jac = _build_bound_inv_logL_and_logL_dB(A_ev, d,
                                                     A_dB_ev, comp_of_dB)
    else:
        fun = lambda x: - logL(A_ev(x), d, invN)
        if A_dB_ev is not None:
            jac = lambda x: - logL_dB(A_ev(x), d, invN, A_dB_ev(x), comp_of_dB)

    # Gather minmize arguments
    if jac is not None:
        minimize_kargs['jac'] = jac
    if disp and 'callback' not in minimize_kargs:
        minimize_kargs['callback'] = verbose_callback()

    # Likelihood maximization
    res = sp.optimize.minimize(fun, *minimize_args, **minimize_kargs)

    # Gather results
    A = A_ev(res.x)
    A_dB = None 
    if A_dB_ev is not None and invN is None:
        res.s, res.invAtNA, res.Sigma = _results_shortcut(
            A, d, A_dB_ev(res.x), comp_of_dB)
    else:
        res.s = _mv(W(A), d)
        res.invAtNA = invAtNA(A, invN)
        fisher = fisher_logL_dB_dB(A, res.s, A_dB_ev(res.x), comp_of_dB, invN)
        res.Sigma = np.linalg.inv(fisher)
    return res


def _results_shortcut(A, d, A_dB, comp_of_dB):
    u, e, v = np.linalg.svd(A, full_matrices=False)
    inv_e = 1 / e
    utd = _mtv(u, d)
    s = _mtv(v, utd * inv_e)
    invAtNA = _mtm(v, inv_e[..., np.newaxis] * v)
    n_param = len(A_dB)
    fisher = np.empty((n_param, n_param))
    x = [_mtv(u, _mv(A_dB_i, s[..., comp_of_dB[i]]))
         for i, A_dB_i in enumerate(A_dB)]
    for i in xrange(n_param):
        for j in xrange(n_param):
            fisher[i, j] = np.sum(x[i] * x[j])
    return s, invAtNA, np.linalg.inv(fisher)


def multi_comp_sep(A_ev, d, invN, patch_ids, *minimize_args, **minimize_kargs):
    """ Perform component separation

    Run an independent `comp_sep` for entries identified by `patch_ids`

    Parameters
    ----------
    A_ev : function or list
        The evaluator of the mixing matrix. It takes a float or an array as
        argument and returns the mixing matrix, a ndarray with shape
        `(..., n_freq, n_comp)`
        If list, the i-th entry is the evaluator of the i-th patch.
    d: ndarray
        The data vector. Shape `(..., n_freq)`.
    invN: ndarray or None
        The inverse noise matrix. Shape `(..., n_freq, n_freq)`.
    patch_ids: array
        id of regions 
    minimize_args: list
        Positional arguments to be passed to `scipy.optimize.minimize`.
        At this moment it just contains `x0`, the initial guess for the spectral
        parameters.
    minimize_kargs: dict
        Keyword arguments to be passed to `scipy.optimize.minimize`.
        Notably, it contains the minimization `method`.

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        Result of the spectral likelihood maximisation
	It is the output of `scipy.optimize.minimize`, and thus includes
	- patch_resx : list
	    the i-th entry is the result of `comp_sep` on `patch_ids == i`
        with the addition of some extra information
	- s : (ndarray)
	    Separated components, collected from all the patches.
            Shape `(..., n_comp)`

    Note
    ----
    The `...` in the arguments denote any extra set of dimention. They have to
    be compatible among different arguments in the `numpy` broadcasting sense.
    """
    # TODO add the possibility of patch specific x0
    assert np.all(patch_ids >= 0)
    max_id = patch_ids.max()

    def patch_comp_sep(patch_id):
        mask = patch_ids == patch_id
        patch_A_ev = A_ev[patch_id] if isinstance(A_ev, list) else A_ev
        return comp_sep(patch_A_ev, d[mask], invN,
                        *minimize_args, **minimize_kargs)

    # Separation
    res = sp.optimize.OptimizeResult()
    res.patch_res = [patch_comp_sep(patch_id) for patch_id in xrange(max_id)]

    # Collect results
    n_comp = res.patch_res[0].s.shape[-1]
    res.s = np.full((d.shape[:-1]+(n_comp,)), np.NaN) # NaN for testing

    for patch_id in xrange(max_id):
        mask = patch_ids == patch_id
        res.s[mask] = res.patch_res[patch_id].s

    return res


def verbose_callback():
    """ Provide a verbose callback function

    NOTE
    ----
    Currently, tested for the bfgs method. It can raise `KeyError` for other
    methods.
    """
    start = time()
    old_old_time = [start]
    old_old_fval = [None]
    def callback(xk):
        k = _get_from_caller('k') + 1
        func_calls = _get_from_caller('func_calls')[0]
        old_fval = _get_from_caller('old_fval')
        old_time = time()
        try:
            logL_message = 'Delta(-logL) = %f' % (old_fval - old_old_fval[0])
        except TypeError:
            logL_message = 'First -logL = %f' % old_fval
        message = [
            'Iter %i' % k,
            'x = %s' % np.array2string(xk),
            logL_message,
            'N Eval = %i' % func_calls,
            'Iter sec = %.2f' % (old_time - old_old_time[0]),
            'Cum sec = %.2f' % (old_time - start),
            ]
        print '\t'.join(message)
        old_old_fval[0] = old_fval
        old_old_time[0] = old_time

    print 'Minimization started'
    return callback


def _get_from_caller(name):
    """ Get the `name` variable from the scope immediately above

    NOTE
    ----
    Kludge for retrieving information inside scipy.optimize.minimize
    """
    caller = inspect.currentframe().f_back.f_back
    return caller.f_locals[name]
