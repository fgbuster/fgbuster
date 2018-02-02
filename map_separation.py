def logL(A_ev, invN, d):
    return lambda x: algebra.logL(A_ev(x), invN, d)

def comp_sep(A_ev, invN, d, *np_minimize_args, **np_minimize_kargs):
    """ Component separation

    A_ev is a function that given a vector of beta computes the mixing matrix
    """
    fun = logL(A_ev, invN, d)
    res = np.minimize(fun, *np_minimize_args, **np_minimize_kargs)
    A = A_ev(res.x)
    W = algebra.W(A, invN)
    res.s = W.dot(d)
    res.invAtNA = # Compute from A and invN
    res.Sigma = # Compute the numerical one? For the analytic we need diff_A
    return res

def multipatch_comp_sep(A_ev, invN, d, patch, *np_minimize_args, **np_minimize_kargs):
    """ Component separation

    Launch comp_sep independently for each index in patch
    """
