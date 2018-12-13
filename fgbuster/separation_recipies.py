""" High-level component separation routines

"""
from six import string_types
import numpy as np
import healpy as hp
from . import algebra as alg
from .mixingmatrix import MixingMatrix


__all__ = [
    'weighted_comp_sep',
    'basic_comp_sep',
]



def weighted_comp_sep(components, instrument, data, cov, nside=0,
                      **minimize_kwargs):
    """ Weighted component separation

    Parameters
    ----------
    components: list or tuple of lists
        List storing the `Components` of the mixing matrix
    instrument:
        Instrument object used to define the mixing matrix.
        It can be any object that has what follows wither as a key or as an
        attribute (e.g. dictionary, PySM.Instrument)
         - Frequencies
    data: ndarray or MaskedArray
        Data vector to be separated. Shape `(n_freq, ..., n_pix)`. `...` can be
        also absent.
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    cov: ndarray or MaskedArray
        Covariance maps. It has to be broadcastable to `data`.
        Notice that you can not pass a pixel independent covariance as an array
        with shape `(n_freq,)`: it has to be `(n_freq, ..., 1)` in order to be
        broadcastable (consider using `basic_comp_sep`, in this case).
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    patch_ids: array
        For each pixel, the array stores the id of the region over which to
        perform component separation independently.

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        See `multi_comp_sep` if `nside` is positive and `comp_sep` otherwise.

    Note
    ----
      * During the component separation, a pixel is masked if at least one of
        its frequencies is masked, either in `data` or in `cov`.
    """
    instrument = _force_keys_as_attributes(instrument)
    # Make sure that cov has the frequency dimension and is equal to n_freq
    cov_shape = list(np.broadcast(cov, data).shape)
    if cov.ndim < 2 or (data.ndim == 3 and cov.shape[-2] == 1):
        cov_shape[-2] = 1
    cov = np.broadcast_to(cov, cov_shape, subok=True)
    
    # Prepare mask and set to zero all the frequencies in the masked pixels: 
    # NOTE: mask are good pixels
    mask = ~(_intersect_mask(data) | _intersect_mask(cov))

    invN = np.zeros(cov.shape[:1] + cov.shape)
    for i in range(cov.shape[0]):
        invN[i, i] = 1. / cov[i]
    invN = invN.T
    if invN.shape[0] != 1:
        invN = invN[mask] 
        
    data_cs = hp.pixelfunc.ma_to_array(data).T[mask]
    assert not np.any(hp.ma(data_cs).mask)

    A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluator(components,
                                                            instrument)
    if not len(x0):
        A_ev = A_ev()

    # Component separation
    if nside:
        patch_ids = hp.ud_grade(np.arange(hp.nside2npix(nside)),
                                hp.npix2nside(data.shape[-1]))[mask]
        res = alg.multi_comp_sep(A_ev, data_cs, invN, A_dB_ev, comp_of_param,
                             patch_ids, x0, **minimize_kwargs)
    else:
        res = alg.comp_sep(A_ev, data_cs, invN, A_dB_ev, comp_of_param, x0,
                       **minimize_kwargs)

    # Craft output
    res.params = params

    def craft_maps(maps):
        # Unfold the masked maps
        # Restore the ordering of the input data (pixel dimension last)
        result = np.full(data.shape[-1:] + maps.shape[1:], hp.UNSEEN)
        result[mask] = maps
        return result.T

    def craft_params(par_array):
        # Add possible last pixels lost due to masking
        # Restore the ordering of the input data (pixel dimension last)
        missing_ids = np.max(patch_ids) - par_array.shape[0] + 1
        extra_dims = np.full((missing_ids,) + par_array.shape[1:], hp.UNSEEN)
        result = np.concatenate((par_array, extra_dims))
        result[np.isnan(result)] = hp.UNSEEN
        return result.T

    if len(x0):
        res.chi_dB = [craft_maps(c) for c in res.chi_dB]
        if nside:
            res.x = craft_params(res.x)
            res.Sigma = craft_params(res.Sigma)

    res.s = craft_maps(res.s)
    res.chi = craft_maps(res.chi)
    res.invAtNA = craft_maps(res.invAtNA)
    res.mask_good = mask

    return res


def basic_comp_sep(components, instrument, data, nside=0, **minimize_kwargs):
    """ Basic component separation

    Parameters
    ----------
    components: list
        List storing the `Components` of the mixing matrix
    instrument
        Instrument object used to define the mixing matrix.
        It can be any object that has what follows either as a key or as an
        attribute (e.g. ``dict``, ``PySM.Instrument``)

         * ``Frequencies``
         * ``Sens_I`` or ``Sens_P`` (optional, frequencies are inverse-noise
           weighted according to these noise levels)

    data: ndarray or MaskedArray
        Data vector to be separated. Shape ``(n_freq, ..., n_pix)``.
        ``...`` can be
          - absent or 1: temperature maps
          - 2: polarization maps
          - 3: temperature and polarization maps (see note)
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    nside:
        For each pixel of a HEALPix map with this nside, the non-linear
        parameters are estimated independently

    Returns
    -------
    result : scipy.optimze.OptimizeResult (dict)
        See `multi_comp_sep` if `nside` is positive and `comp_sep` otherwise.

    Note
    ----
      * During the component separation, a pixel is masked if at least one of
        its frequencies is masked.
      * If you provide temperature and polarization maps, they will constrain the
        **same** set of parameters. In particular, separation is **not** done
        independently for temperature and polarization. If you want an
        independent fitting for temperature and polarization, please launch

         res_T = basic_comp_sep(component_T, instrument, data[:, 0], **kwargs)
         res_P = basic_comp_sep(component_P, instrument, data[:, 1:], **kwargs)

    """
    instrument = _force_keys_as_attributes(instrument)
    # Prepare mask and set to zero all the frequencies in the masked pixels:
    # NOTE: mask are bad pixels
    mask = _intersect_mask(data)
    data = hp.pixelfunc.ma_to_array(data).copy()
    data[..., mask] = 0  # Thus no contribution to the spectral likelihood

    try:
        data_nside = hp.get_nside(data[0])
    except TypeError:
        data_nside = 0
    prewhiten_factors = _get_prewhiten_factors(instrument, data.shape,
                                               data_nside)
    A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluator(
        components, instrument, prewhiten_factors=prewhiten_factors)
    if not len(x0):
        A_ev = A_ev()
    if prewhiten_factors is None:
        prewhitened_data = data.T
    else:
        prewhitened_data = prewhiten_factors * data.T

    # Component separation
    if nside:
        patch_ids = hp.ud_grade(np.arange(hp.nside2npix(nside)),
                                hp.npix2nside(data.shape[-1]))
        res = alg.multi_comp_sep(
            A_ev, prewhitened_data, None, A_dB_ev, comp_of_param, patch_ids,
            x0, **minimize_kwargs)
    else:
        res = alg.comp_sep(A_ev, prewhitened_data, None, A_dB_ev, comp_of_param,
                           x0, **minimize_kwargs)

    # Craft output
    # 1) Apply the mask, if any
    # 2) Restore the ordering of the input data (pixel dimension last)
    res.params = params
    res.s = res.s.T
    res.s[..., mask] = hp.UNSEEN
    res.chi = res.chi.T
    res.chi[..., mask] = hp.UNSEEN
    if 'chi_dB' in res:
        for i in range(len(res.chi_dB)):
            res.chi_dB[i] = res.chi_dB[i].T
            res.chi_dB[i][..., mask] = hp.UNSEEN
    if nside and len(x0):
        x_mask = hp.ud_grade(mask.astype(float), nside) == 1.
        res.x[x_mask] = hp.UNSEEN
        res.Sigma[x_mask] = hp.UNSEEN
        res.x = res.x.T
        res.Sigma = res.Sigma.T
    return res


def _get_prewhiten_factors(instrument, data_shape, nside):
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
        else:
            raise ValueError(data_shape)
    except AttributeError:  # instrument has no sensitivity -> do not prewhite
        return None

    assert np.all(np.isfinite(sens))
    if nside:
        return hp.nside2resol(nside, arcmin=True) / sens
    else:
        return 12**0.5 * hp.nside2resol(1, arcmin=True) / sens


def _A_evaluator(components, instrument, prewhiten_factors=None):
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument.Frequencies)
    A_dB_ev = A.diff_evaluator(instrument.Frequencies)
    comp_of_dB = A.comp_of_dB
    x0 = np.array([x for c in components for x in c.defaults])
    params = A.params

    if prewhiten_factors is None:
        return A_ev, A_dB_ev, comp_of_dB, x0, params

    if A.n_param:
        pw_A_ev = lambda x: prewhiten_factors[..., np.newaxis] * A_ev(x)
        pw_A_dB_ev = lambda x: [prewhiten_factors[..., np.newaxis] * A_dB_i
                                for A_dB_i in A_dB_ev(x)]
    else:
        pw_A_ev = lambda: prewhiten_factors[..., np.newaxis] * A_ev()
        pw_A_dB_ev = None

    return pw_A_ev, pw_A_dB_ev, comp_of_dB, x0, params


def _intersect_mask(maps):
    if hp.pixelfunc.is_ma(maps):
        mask = maps.mask
    else:
        mask = maps == hp.UNSEEN

    # Mask entire pixel if any of the frequencies in the pixel is masked
    return np.any(mask, axis=tuple(range(maps.ndim-1)))


# What the hell are _LowerCaseAttrDict and _force_keys_as_attributes?
# Why are you so twisted?!? Because we decided that instrument can be either a
# PySM.Instrument or a dictionary (especially the one used to construct a
# PySM.Instrument).
#
# Suppose I want the frequencies.
# * Pysm.Instrument 
#     freqs = instrument.Frequencies
# * the configuration dict of a Pysm.Instrument
#     freqs = instrument['frequencies']
# We force the former API in the dictionary case by
# * setting all string keys to lower-case
# * making dictionary entries accessible as attributes
# * Catching upper-case attribute calls and returning the lower-case version
# Shoot us any simpler idea. Maybe dropping the PySM.Instrument support...
class _LowerCaseAttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(_LowerCaseAttrDict, self).__init__(*args, **kwargs)
        for key, item in self.items():
            if isinstance(key, string_types):
                str_key = str(key)
                if str_key.lower() != str_key:
                    self[str_key.lower()] = item
                    del self[key]
        self.__dict__ = self

    def __getattr__(self, key):
        try:
            return self[key.lower()]
        except KeyError:
            raise AttributeError("No attribute named '%s'" % key)


def _force_keys_as_attributes(instrument):
    if hasattr(instrument, 'Frequencies'):
        return instrument
    else:
        return _LowerCaseAttrDict(instrument)
