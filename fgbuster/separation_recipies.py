""" Component separation with many different setups

"""
import numpy as np
import healpy as hp
from .algebra import multi_comp_sep, comp_sep
from .mixingmatrix import MixingMatrix

def basic_comp_sep(components, instrument, data, nside=0, **minimize_kwargs):
    """ Basic component separation

    Parameters
    ----------
    components: list or tuple of lists
        Two input format are allowed
         - List `Components` of the mixing matrix
         - Tuple with two lists of `Components`, the first for temperature and
           the second for polarization
    instrument: PySM.Instrument
        Instrument object used to define the mixing matrix and the
        frequency-dependent noise weight.
        It is required to have:
         - frequencies
        however, also the following are taken into account, if provided
         - sens_I or sens_P (define the frequency inverse noise)
         - bandpass (the mixing matrix is integrated over the bandpass)
    data: ndarray or MaskedArray
        Data vector to be separated. Shape (n_freq, ..., n_pix)
        If `...` is 2, use sens_P to define the weights, sens_I otherwise.
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process. 
        Note that a pixel is masked if at least one of its frequencies is masked
    nside:
        For each pixel of a HEALPix map with this nside, the non-linear
        parameters are estimated independently

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        See `multi_comp_sep` if `nside` is positive and `comp_sep` otherwise.

    """
    # Prepare mask and set to zero all the frequencies in the masked pixels: 
    mask = _intersect_mask(data)
    data = hp.pixelfunc.ma_to_array(data).copy()
    data[..., mask] = 0  # Thus no contribution to the spectral likelihood

    prewhiten_factors = _get_prewhiten_factors(instrument, data.shape)
    A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(
        components, instrument, prewhiten_factors=prewhiten_factors)
    prewhitened_data = prewhiten_factors * data.T

    # Launch component separation
    if nside == 0:
        res = comp_sep(A_ev, prewhitened_data, None, A_dB_ev, comp_of_param, x0,
                       **minimize_kwargs)
    else:
        patch_ids = hp.ud_grade(np.arange(hp.nside2npix(nside)),
                                hp.npix2nside(data.shape[-1]))
        res = multi_comp_sep(
            A_ev, prewhitened_data, None, A_dB_ev, comp_of_param, patch_ids,
            x0, **minimize_kwargs)

    # Craft output
    # 1) Apply the mask, if any
    # 2) Restore the ordering of the input data (pixel dimension last)
    res.params = params
    res.s = res.s.T
    res.s[..., mask] = hp.UNSEEN
    res.chi = res.chi.T
    res.chi[..., mask] = hp.UNSEEN
    if nside:
        x_mask = hp.ud_grade(mask.astype(float), nside) == 1.
        res.x[x_mask] = hp.UNSEEN
        res.Sigma[x_mask] = hp.UNSEEN
        res.x = res.x.T
        res.Sigma = res.Sigma.T
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
    try:
        if len(data_shape) < 3 or data_shape[1] == 1:
            sens = instrument.Sens_I
        elif data_shape[1] == 2:
            sens = instrument.Sens_P
        elif data_shape[1] == 3:
            sens = np.stack(
                (instrument.Sens_I, instrument.Sens_P, instrument.Sens_P))
    except AttributeError:  # instrument has no sensitivity -> do not prewhite
        print('The sensitivity of the instrument is not specified')
        return None

    assert np.all(np.isfinite(sens))
    return hp.nside2resol(instrument.Nside, arcmin=True) / sens


def _A_evaluators(components, instrument, prewhiten_factors=None):
    if hp.cookbook.is_seq_of_seq(components):
        return _T_P_A_evaluators(components, instrument, prewhiten_factors)
    return _single_A_evaluators(components, instrument, prewhiten_factors)


def _T_P_A_evaluators(components, instrument, prewhiten_factors=None):
    A_ev_T, A_dB_ev_T, comp_of_dB_T, x0_T, params_T = _A_evaluators(
        components[0], instrument)
    A_ev_P, A_dB_ev_P, comp_of_dB_P, x0_P, params_P = _A_evaluators(
        components[1], instrument)

    def A_ev(x):
        A_T = A_ev_T(x[:len(params_T)])
        A_P = A_ev_P(x[len(params_T):])
        return np.stack((A_T, A_P, A_P))

    def A_dB_ev(x):
        A_dB_T = A_dB_ev_T(x[:len(params_T)])
        A_dB_P = A_dB_ev_P(x[len(params_T):])
        return A_dB_T + A_dB_P

    comp_of_dB = [(el,) + (0,) + (c,) for el, c in comp_of_dB_T]
    comp_of_dB += [(el, slice(1, 3, None), c) for el, c in comp_of_dB_P]
    x0 = np.hstack((x0_T, x0_P))
    params = ['T.%s' % p for p in params_T] + ['P.%s' % p for p in params_P]
    if prewhiten_factors is None:
        return A_ev, A_dB_ev, comp_of_dB, x0, params

    pw_A_ev = lambda x: prewhiten_factors[..., np.newaxis] * A_ev(x)
    if len(prewhiten_factors.shape) < 2 or prewhiten_factors.shape[-2] < 2:
        # prewhiten_factors is not stokes-dependent
        pwf_dB = [prewhiten_factors] * len(params_T)
    else:
        pwf_dB = [prewhiten_factors[..., 0, :, np.newaxis]] * len(params_T)
        pwf_dB += [prewhiten_factors[..., 1:, :, np.newaxis]] * len(params_P)
    pw_A_dB_ev = lambda x: [pwf_dB_i * A_dB_i
                            for pwf_dB_i, A_dB_i in zip(pwf_dB, A_dB_ev(x))]
    return pw_A_ev, pw_A_dB_ev, comp_of_dB, x0, params


def _single_A_evaluators(components, instrument, prewhiten_factors=None):
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument.Frequencies)
    A_dB_ev = A.diff_evaluator(instrument.Frequencies)
    comp_of_dB = A.comp_of_dB
    x0 = np.array([x for c in components for x in c.defaults])
    params = A.params
    if prewhiten_factors is None:
        return A_ev, A_dB_ev, comp_of_dB, x0, params

    pw_A_ev = lambda x: prewhiten_factors[..., np.newaxis] * A_ev(x)
    pw_A_dB_ev = lambda x: [prewhiten_factors[..., np.newaxis] * A_dB_i
                            for A_dB_i in A_dB_ev(x)]
    return pw_A_ev, pw_A_dB_ev, comp_of_dB, x0, params

def _intersect_mask(maps):
    if hp.pixelfunc.is_ma(maps):
        mask = maps.mask
    else:
        mask = maps == hp.UNSEEN

    # Mask entire pixel if any of the frequencies in the pixel is masked
    return np.any(mask, axis=tuple(range(maps.ndim-1)))
