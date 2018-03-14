""" Component separation with many different setups

"""
import numpy as np
import healpy as hp
from algebra import multi_comp_sep, comp_sep

def basic_comp_sep(components, instrument, data, nside=0):
    """ Basic component separation

    Parameters
    ----------
    components: list
        List of the `Components` in the data model
    instrument: PySM.Instrument
        Instrument object used to define the mixing matrix and the
        frequency-dependent noise weight.
        It is required to have:
        - frequencies
        however, also the following are taken into account, if provided
        - sens_I or sens_P (define the frequency inverse noise)
        - bandpass (the mixing matrix is integrated over the bandpass)
    data: array
        Data vector to be separated. Shape (n_freq, ..., n_pix)
        If `...` is 2, use sens_P to define the weights, sens_I otherwise.
    nside:
        For each pixel of a HEALPix map with this nside, the non-linear
        parameters are estimated independently

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        see `multi_comp_sep`
    """
    # TODO handle temperature and polarization jointly

    prewhiten_factors = _get_prewhiten_factors(instrument, data.shape)
    A_ev, A_dB_ev, comp_of_param, params = _build_A_evaluators(
        components, instrument, prewhiten_factors=prewhiten_factors)
    x0 = np.array([x for c in components for x in c.defaults])
    prewhitened_data = prewhiten_factors * data.T
    if nside == 0:
        res = comp_sep(A_ev, prewhitened_data, None, A_dB_ev, comp_of_param, x0,
                       options=dict(disp=True))
    else:
        patch_ids = hp.ud_grade(np.arange(hp.nside2npix(nside)),
                                hp.npix2nside(data.shape[-1]))
        res = multi_comp_sep(A_ev, prewhitened_data, None, patch_ids, x0)

    # Launch component separation
    res.params = params
    res.s = res.s.T
    return res


def _get_prewhiten_factors(instrument, data_shape):
    """ Derive the prewhitening factor from the sensitivity

    Parameters
    ----------
    instrument: PySM.Instrument
    data_shape: tuple
        It is expected to be `(n_freq, n_stokes, n_pix)`. `n_stokes` is used to
        define if sens_I or sens_P (or both) should be used to compute the
        factors.
        - If `n_stokes` is absent or `n_stokes == 1`, use sens_I.
        - If `n_stokes == 2`, use sens_P.
        - If `n_stokes == 3`, the factors will have shape (3, n_freq). Sens_I is
          used for [0, :], while sens_P is used for [1:, :].

    Returns
    -------
    factor: array
        prewhitening factors
    """
    # TODO handle temperature and polarization jointly
    try:
        if len(data_shape) < 3 or data_shape[1] == 1:
            sens = instrument.Sens_I
        elif data_shape[1] == 2:
            sens = instrument.Sens_P
        elif data_shape[1] == 3:
            sens = np.stack(
                (instrument.Sens_I, instrument.Sens_P, instrument.Sens_P))
    except AttributeError:  # instrument has no sensitivity -> do not prewhite
        print 'The sensitivity of the instrument is not specified'
        return None

    return hp.nside2resol(instrument.Nside, arcmin=True) / sens


def _build_A_evaluators(components, instrument, prewhiten_factors=None):
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument.Frequencies)
    A_dB_ev = A.gradient_evaluator(instrument.Frequencies)
    if prewhiten_factors is None:
        return A_ev, A_dB_ev, A.comp_of_param, A.params
    else:
        pw_A_ev =  lambda x: prewhiten_factors[..., np.newaxis] * A_ev(x)
        pw_A_dB_ev =  lambda x: [prewhiten_factors[..., np.newaxis] * A_dB_i
                                 for A_dB_i in A_dB_ev(x)]
        return pw_A_ev, pw_A_dB_ev, A.comp_of_param, A.params


class MixingMatrix(tuple):
    """ Collection of Components

    The goal is to provide ways to evaluate all the components (or their
    derivatives) with a single call and store them in a matrix (the mixing
    matrix).

    There are two ways:
    - evaluate it using (nu, param_0, param_1, param_2, ...)
    - provide A_ev, which takes a single array as argument
    """
    # XXX if we plan on using just the second approach this class is a wash
    # and should be removed.

    def __new__(cls, *components):
        return tuple.__new__(cls, components)

    def __init__(self, *components):
        super(MixingMatrix, self).__init__(*components)
        self.__first_param_of_comp = []
        self.__comp_of_param = []
        for i_c, c in enumerate(components):
            self.__first_param_of_comp.append(self.n_param)
            self.__comp_of_param += [i_c] * c.n_param

    @property
    def params(self):
        # TODO: handle components with the same name
        return ['%s.%s' % (type(c).__name__, p)
                for c in self for p in c.params]

    @property
    def n_param(self):
        return len(self.__comp_of_param)

    @property
    def comp_of_param(self):
        return self.__comp_of_param

    def eval(self, nu, *params):
        shape = np.broadcast(*params).shape + (len(nu), len(self))
        res = np.zeros(shape)
        for i_c, c in enumerate(self):
            i_fp = self.__first_param_of_comp[i_c]
            res[..., i_c] += c.eval(nu, *params[i_fp: i_fp + c.n_param])
        return res

    def evaluator(self, nu, shape=(-1,)):
        def f(param_array):
            param_array = np.array(param_array)
            return self.eval(nu, *[p for p in param_array.reshape(shape)])
        return f

    def gradient(self, nu, *params):
        if not params:
            return None
        shape = (len(params),) + np.broadcast(*params).shape + (len(nu),)
        res = []
        for i_c, c in enumerate(self):
            param_slice = slice(self.__first_param_of_comp[i_c],
                                self.__first_param_of_comp[i_c] + c.n_param)
            res += [g.reshape(-1, 1) 
                    for g in c.gradient(nu, *params[param_slice])]
        return res

    def gradient_evaluator(self, nu, shape=(-1,)):
        def f(param_array):
            param_array = np.array(param_array)
            return self.gradient(nu, *[p for p in param_array.reshape(shape)])
        return f
