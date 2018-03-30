""" Recurrent algebraic functions in component separation

"""

# Note for developpers
# --------------------
# The functions in this file conform to the following convetions
# 1) Matrices are ndarrays. The last two dimensions are respectively the
#    codomain and the domain, the other dimensions identify the block on the
#    diagonal. For example, an M-by-N matrix that is block-diagonal, with K
#    diagonal blocks, is represented as an ndarray with shape (K, M/K, N/K).
# 2) Matrices can have multiple indices for the diagonal blocks. For
#    instance, if in the previous example K = JxL, the shape of the ndarray is
#    (J, L, M/K, N/K).
# 3) Suppose that the blocks are equal for the same index in J and different
#    index in L, the matrix can be passed as a (K, 1, M/K, M/K) ndarray, without
#    repeating equal blocks.
# 4) Vectors are just like matrices, without the domain dimension
# 5) Many functions come in pairs foo(A, invN, ...) and _foo_svd(u_e_v, ...)
#    In _foo_svd does what foo is supposed to, but:
#     - instead of providing A you provide its SVD, u_e_v
#     - the domain of input and outputs matrices and vectors is prewhitend with
#       sqrt(invN)
#     - toto can return the SVD, which can then be reused in _bar_svd(...)

import inspect
from time import time
import numpy as np
import scipy as sp
import numdifftools

OPTIMIZE = False
_EPSILON_LOGL_DB = 1e-6


def _inv(m):
    result = np.array(map(np.linalg.inv, m.reshape((-1,)+m.shape[-2:])))
    return result.reshape(m.shape)


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

def _svd_sqrt_invN_A(A, invN=None, L=None):
    """ SVD of A and Cholesky factor of invN

    Prewhiten `A` according to `invN` (if either `invN` of `L` is provided) and
    return both its SVD and the Cholesky factor of `invN`.
    If you provide the Cholesky factor L, invN is ignored.
    """

    if L is None and invN is not None:
        L = np.linalg.cholesky(invN)

    if L is not None:
        A = _mtm(L, A)

    u_e_v = np.linalg.svd(A, full_matrices=False)
    return u_e_v, L


def _logL_svd(u_e_v, d):
    return 0.5 * np.linalg.norm(_mtv(u_e_v[0], d))**2


def logL(A, d, invN=None, return_svd=False):
    try:
        u_e_v, L = _svd_sqrt_invN_A(A, invN)
    except np.linalg.linalg.LinAlgError:
        print 'SVD of A failed -> logL = -inf'
        return - np.inf

    if L is not None:
        d = _mtv(L, d)
    res = _logL_svd(u_e_v, d)

    if return_svd:
        return res, (u_e_v, L)
    return res


def _invAtNA_svd(u_e_v):
    _, e, v = u_e_v
    return _mtm(v, v / e[..., np.newaxis]**2)


def invAtNA(A, invN=None, return_svd=False):
    u_e_v, L = _svd_sqrt_invN_A(A, invN)
    res = _invAtNA_svd(u_e_v)
    if return_svd:
        return res, (u_e_v, L)
    return res


def _As_svd(u_e_v, s):
    u, e, v = u_e_v
    return _mv(u, e * _mv(v, s))


def _Wd_svd(u_e_v, d):
    u, e, v = u_e_v
    utd = _mtv(u, d)
    return _mtv(v, utd / e)


def Wd(A, d, invN=None, return_svd=False):
    u_e_v, L = _svd_sqrt_invN_A(A, invN)
    if L is not None:
        d = _mtv(L, d)
    res = _Wd_svd(u_e_v, d)
    if return_svd:
        return res, (u_e_v, L)
    return res


def _W_svd(u_e_v):
    u, e, v = u_e_v
    return _mtm(v, _T(u) / e[..., np.newaxis])


def W(A, invN=None, return_svd=False):
    u_e_v, L = _svd_sqrt_invN_A(A, invN)
    if L is None:
        res = _W_svd(u_e_v)
    else:
        res = _mm(_W_svd(u_e_v), _T(L))
    if return_svd:
        return res, (u_e_v, L)
    return res


def W_dB(A, A_dB, comp_of_dB, invN=None):
   """ Derivative of W
    which could be particularly useful for the computation of residuals
    through the first order development of the map-making equation

    Parameters
    ----------
    A : ndarray
        Mixing matrix. Shape `(..., n_freq, n_comp)`
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
    res : array
        Derivative of W. If `A_dB` is a list, `res[i]`
        is computed from `A_dB[i]`.
    """
    AtNAinv = invAtNA(A, invN=invN)
    A_dB, comp_of_dB = _A_dB_and_comp_of_dB_as_compatible_list(A_dB, comp_of_dB)

    n_param = len(A_dB)
    res = []
    for i in xrange(n_param):
        dAtNAinv_ = -_mm(_T(_mm(A_dB[i],AtNAinv)),  _mm(invN, A))
        dAtNAinv = dAtNAinv_ + _T(dAtNAinv_)
        res_a = _mm(dAtNAinv, _mtm(A,invN))
        res_b = _mm(AtNAinv, _mtm(A_dB[i], invN))
        res.append(res_a+res_b)
    return res


def W_dB_dB(A, A_dB, A_dBdB, comp_of_dB, invN=None):
   """ Second Derivative of W
    which could be particularly useful for the computation of 
    *statistical* residuals through the second order development 
    of the map-making equation

    Parameters
    ----------
    A : ndarray
        Mixing matrix. Shape `(..., n_freq, n_comp)`
    invN: ndarray or None
        The inverse noise matrix. Shape `(..., n_freq, n_freq)`.
    A_dB : ndarray or list of ndarray
        The derivative of the mixing matrix. If list, each entry is the
        derivative with respect to a different parameter.
   A_dBdB : ndarray or list of ndarray
        The second derivative of the mixing matrix. If list, each entry is the
        derivative with respect to a different parameter.
    comp_of_dB: index or list of indices
        It allows to provide in `A_dB` only the non-zero columns `A`.
        `A_dB` is assumed to be the derivative of `A[..., comp_of_dB]`.
        If a list is provided, also `A_dB` has to be a list and
        `A_dB[i]` is assumed to be the derivative of `A[..., comp_of_dB[i]]`.

    Returns
    -------
    res : array
        Second Derivative of W. If `A_dB` is a list, `res[i]`
        is co
    """
    raise NotImplementedError
    

def logL_dB(A, d, invN, A_dB, comp_of_dB=np.s_[...], return_svd=False):
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
    comp_of_dB: IndexExpression or list of IndexExpression
        It allows to provide in `A_dB` only the non-zero columns `A`.
        `A_dB` is assumed to be the derivative of `A[comp_of_dB]`.
        If a list is provided, also `A_dB` has to be a list and
        `A_dB[i]` is assumed to be the derivative of `A[comp_of_dB[i]]`.

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
    A_dB, comp_of_dB = _A_dB_and_comp_of_dB_as_compatible_list(A_dB, comp_of_dB)

    u_e_v, L = _svd_sqrt_invN_A(A, invN)
    if L is not None:
        A_dB = [_mtm(L, A_dB_i) for A_dB_i in A_dB]
        d = _mtv(L, d)
    res = _logL_dB_svd(u_e_v, d, A_dB, comp_of_dB)
    if return_svd:
        return res, (u_e_v, L)
    return res


def _A_dB_and_comp_of_dB_as_compatible_list(A_dB, comp_of_dB):
    if not isinstance(A_dB, list):
        A_dB = [A_dB]

    if isinstance(comp_of_dB, list):
        assert len(A_dB) == len(comp_of_dB)
    else:
        comp_of_dB = [comp_of_dB] * len(A_dB)

    # The following ensures that s[comp_of_dB[i]] still has all the axes
    comp_of_dB = [_turn_into_slice_if_integer(c) for c in comp_of_dB]

    return A_dB, comp_of_dB


def _turn_into_slice_if_integer(index_expression):
    # When you index an array with an integer you lose one dimension.
    # To avoid this we turn the integer into a slice
    res = []
    for i in index_expression:
        if isinstance(i, (int, long)):
            res.append(slice(i, i+1, None))
        else:
            res.append(i)
    return tuple(res)


def _A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_dB, x):
    # XXX: It can be expansive. Make the user responsible for these checks?
    if A_dB_ev is None:
        return None, None
    A_dB = A_dB_ev(x)
    if not isinstance(A_dB, list):
        A_dB_ev = lambda x: [A_dB_ev(x)]
        A_dB = [A_dB]

    if isinstance(comp_of_dB, list):
        assert len(A_dB) == len(comp_of_dB)
    else:
        comp_of_dB = [comp_of_dB] * len(A_dB)

    # The following ensures that s[comp_of_dB[i]] still has all the axes
    comp_of_dB = [_turn_into_slice_if_integer(c) for c in comp_of_dB]

    return A_dB_ev, comp_of_dB


def _fisher_logL_dB_dB_svd(u_e_v, s, A_dB, comp_of_dB):
    u, _, _ = u_e_v
    x = []
    for i in xrange(len(A_dB)):
        D_A_dB_s = np.zeros(s.shape[:-1] + u.shape[-2:-1])  # Full shape
        comp_freq_of_dB = comp_of_dB[i][:-1] + (slice(None),)
        comp_freq_of_dB += comp_of_dB[i][-1:]
        A_dB_s = _mv(A_dB[i], s[comp_of_dB[i]])  # Compressed shape
        D_A_dB_s[comp_freq_of_dB[:-1]] = (
            A_dB_s - _mv(u[comp_freq_of_dB], _mtv(u[comp_freq_of_dB], A_dB_s)))
        x.append(D_A_dB_s)

    return np.array([[np.sum(x_i*x_j) for x_i in x] for x_j in x])


def fisher_logL_dB_dB(A, s, A_dB, comp_of_dB, invN=None, return_svd=False):
    A_dB, comp_of_dB = _A_dB_and_comp_of_dB_as_compatible_list(A_dB, comp_of_dB)
    u_e_v, L = _svd_sqrt_invN_A(A, invN)
    if L is not None:
        A_dB = [_mtm(L, A_dB_i) for A_dB_i in A_dB]
    res = _fisher_logL_dB_dB_svd(u_e_v, s, A_dB, comp_of_dB)
    if return_svd:
        return res, (u_e_v, L)
    return res


def _build_bound_inv_logL_and_logL_dB(A_ev, d, invN,
                                      A_dB_ev=None, comp_of_dB=None):
    # XXX: Turn this function into a class?
    """ Produce the functions -logL(x) and -logL_dB(x)

    Keep in the memory the the quantities computed for the last value of x.
    If x of the next call coincide with the last one, recycle the pre-computed
    quantities. It gives ~2x speedup if you often compute both -logL and
    -logL_dB for the same x.
    """
    L = [None]
    x_old = [None]
    u_e_v_old = [None]
    A_dB_old = [None]
    inv_e_old = [None]
    pw_d = [None]

    def _update_old(x):
        # If x is different from the last one, update the SVD
        if not np.all(x == x_old[0]):
            u_e_v_old[0], L[0] = _svd_sqrt_invN_A(A_ev(x), invN, L[0])
            inv_e_old[0] = 1. / u_e_v_old[0][1]
            if A_dB_ev is not None:
                if L[0] is None:
                    A_dB_old[0] = A_dB_ev(x)
                else:
                    A_dB_old[0] = [_mtm(L[0], A_dB_i) for A_dB_i in A_dB_ev(x)]
            x_old[0] = x
            if pw_d[0] is None:  # If this is the first call, prewhiten d
                if L[0] is None:
                    pw_d[0] = d
                else:
                    pw_d[0] = _mtv(L[0], d)

    def _inv_logL(x):
        _update_old(x)
        return - _logL_svd(u_e_v_old[0], pw_d[0])

    if A_dB_ev is None:
        def _inv_logL_dB(x):
            return sp.optimize.approx_fprime(x, _inv_logL, _EPSILON_LOGL_DB)
    else:
        def _inv_logL_dB(x):
            _update_old(x)
            return - _logL_dB_svd(u_e_v_old[0], pw_d[0],
                                  A_dB_old[0], comp_of_dB)

    return _inv_logL, _inv_logL_dB, (u_e_v_old, A_dB_old, x_old, pw_d)


def comp_sep(A_ev, d, invN, A_dB_ev, comp_of_dB,
             *minimize_args, **minimize_kwargs):
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
    comp_of_dB: list of IndexExpression
        It allows to provide as output of `A_dB_ev` only the non-zero columns
        `A`. `A_dB_ev(x)[i]` is assumed to be the derivative of
        `A[comp_of_dB[i]]`.
    minimize_args: list
        Positional arguments to be passed to `scipy.optimize.minimize`.
        At this moment it just contains `x0`, the initial guess for the spectral
        parameters
    minimize_kwargs: dict
        Keyword arguments to be passed to `scipy.optimize.minimize`.
        A good choice for most cases is
        `minimize_kwargs = {'tol': 1, options: {'disp': True}}`. `tol` depends
        on both the solver and your signal to noise: it should ensure that the
        difference between the best fit -logL and and the minimum is well less
        then 1, without exagereting (a difference of 1e-4 is useless).
        `disp` also triggers a verbose callback that monitors the convergence.

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        Result of the spectral likelihood maximisation
        It is the output of `scipy.optimize.minimize`, plus some extra.
        It includes
        - x : (array)
            Maximum likelihood spectral parameters
        - Sigma : (ndarray)
            Covariance of the spectral parameters,
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
    if A_dB_ev is not None:
        A_dB_ev, comp_of_dB = _A_dB_ev_and_comp_of_dB_as_compatible_list(
            A_dB_ev, comp_of_dB, minimize_args[0])
    disp = 'options' in minimize_kwargs and 'disp' in minimize_kwargs['options']

    # Prepare functions for minimize
    fun, jac, last_values = _build_bound_inv_logL_and_logL_dB(
        A_ev, d, invN, A_dB_ev, comp_of_dB)
    minimize_kwargs['jac'] = jac

    # Gather minmize arguments
    if disp and 'callback' not in minimize_kwargs:
        minimize_kwargs['callback'] = verbose_callback()

    # Likelihood maximization
    res = sp.optimize.minimize(fun, *minimize_args, **minimize_kwargs)

    # Gather results
    u_e_v_last, A_dB_last, x_last, pw_d = last_values
    if not np.all(x_last[0] == res.x):
        fun(res.x) #  Make sure that last_values refer to the minimum

    res.s = _Wd_svd(u_e_v_last[0], pw_d[0])
    res.invAtNA = _invAtNA_svd(u_e_v_last[0])
    res.chi = pw_d[0] - _As_svd(u_e_v_last[0], res.s)
    if A_dB_ev is None:
        fisher = numdifftools.Hessian(fun)(res.x)  # TODO: something cheaper
    else:
        fisher = _fisher_logL_dB_dB_svd(u_e_v_last[0], res.s,
                                        A_dB_last[0], comp_of_dB)
        As_dB = (_mv(A_dB_i, res.s[comp_of_dB_i])
                 for A_dB_i, comp_of_dB_i in zip(A_dB_last[0], comp_of_dB))
        res.chi_dB = []
        for comp_of_dB_i, As_dB_i in zip(comp_of_dB, As_dB):
            freq_of_dB = comp_of_dB_i[:-1] + (slice(None),)
            res.chi_dB.append(np.sum(res.chi[freq_of_dB] * As_dB_i, -1)
                              / np.linalg.norm(As_dB_i, axis=-1))
    res.Sigma = np.linalg.inv(fisher)
    return res


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
        id of regions.
    minimize_args: list
        Positional arguments to be passed to `scipy.optimize.minimize`.
        At this moment it just contains `x0`, the initial guess for the spectral
        parameters.
    minimize_kwargs: dict
        Keyword arguments to be passed to `scipy.optimize.minimize`.
        A good choice for most cases is
        `minimize_kwargs = {'tol': 1, options: {'disp': True}}`. `tol` depends
        on both the solver and your signal to noise: it should ensure that the
        difference between the best fit -logL and the minimum is way less
        than 1, without exagerating (a difference of 1e-4 is useless).
        `disp` also triggers a verbose callback that monitors the convergence.

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        Result of the spectral likelihood maximization
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
    # TODO: add the possibility of patch specific x0
    # TODO: mask input arrays. What about masking where patch_ids < 0?
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
    Kludge for retrieving information from inside scipy.optimize.minimize
    """
    caller = inspect.currentframe().f_back.f_back
    return caller.f_locals[name]
