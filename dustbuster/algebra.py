""" Recurrent algebraic functions in component separation

"""
import numpy as np
import scipy as sp
import numdifftools as nd

OPTIMIZE = False


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

def logL(A, d, invN=None):
    if invN is None:
        Ad = _mtv(A, d)
        return np.sum(Ad * _mv(_inv(_mtm(A, A)), Ad))
    ANd = _mtmv(A, invN, d)
    return np.sum(ANd * mv(_inv(_mtmm(A, invN, A)), ANd))


def W(A, invN=None):
    if invN is None:
        invAA = _inv(_mtm(A, A))
        return _mm(invAA, _T(A))
    invAA = _inv(_mtmm(A, invN, A))
    return _mmm(invAA, _T(A), invN)


def invAtNA(A, invN=None):
    if invN is None:
        return _inv(_mtm(A, A))
    return _inv(_mtmm(A, invN, A))


def W_dB(A, diff_A, invN=None):
    raise NotImplementedError


def W_dB_dB(A, invN=None):
    raise NotImplementedError


def logL_dB(A, s, invN=None):
    raise NotImplementedError


def logL_dB_dB(A, s, diff_A, invN=None):
    raise NotImplementedError


def comp_sep(A_ev, d, invN, *minimize_args, **minimize_kargs):
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
    fun = lambda x: - logL(A_ev(x), d, invN)
    res = sp.optimize.minimize(fun, *minimize_args, **minimize_kargs)
    A = A_ev(res.x)
    res.s = _mv(W(A, invN), d)
    res.invAtNA = invAtNA(A, invN)
    res.Sigma = 2 * _inv(nd.Hessian(fun)(res.x))
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
