import numdifftools as nd
import algebra

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
    invN: ndarray
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
    -----
    The `...` in the arguments denote any extra set of dimention. They have to
    be compatible among different arguments in the `numpy` broadcasting sense.
    """
    fun = lambda x: - algebra.logL(A_ev(x), d, invN)
    res = np.minimize(fun, *np_minimize_args, **np_minimize_kargs)
    A = A_ev(res.x)
    W = algebra.W(A, invN)
    res.s = W.dot(d)
    res.invAtNA = algebra.invAtNA(A, invN)
    res.Sigma = 2 * inv(nd.Hessian(fun)(res.x))
    return res


def multi_comp_sep(A_ev_list, d, invN, patch_id,
                   *minimize_args, **minimize_kargs):
