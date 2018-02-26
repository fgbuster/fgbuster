""" Recurrent algebraic functions in component separation

"""
import numpy as np

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
