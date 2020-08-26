# FGBuster
# Copyright (C) 2019 Davide Poletti, Josquin Errard and the FGBuster developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" High-level component separation routines

"""
from six import string_types
import numpy as np
from scipy.optimize import OptimizeResult
import healpy as hp
from . import algebra as alg
from .mixingmatrix import MixingMatrix


__all__ = [
    'basic_comp_sep',
    'weighted_comp_sep',
    'ilc',
    'harmonic_ilc',
]



def weighted_comp_sep(components, instrument, data, cov, nside=0,
                      **minimize_kwargs):
    """ Weighted component separation

    Parameters
    ----------
    components: list or tuple of lists
        List storing the :class:`Component` s of the mixing matrix
    instrument:
        Instrument used to define the mixing matrix.
        It can be any object that has what follows either as a key or as an
        attribute (e.g. `dict`, `PySM.Instrument`)

        - **Frequencies**

    data: ndarray or MaskedArray
        Data vector to be separated. Shape *(n_freq, ..., n_pix)*. *...* can be
        also absent.
        Values equal to `hp.UNSEEN` or, if `MaskedArray`, masked values are
        neglected during the component separation process.
    cov: ndarray or MaskedArray
        Covariance maps. It has to be broadcastable to *data*.
        Notice that you can not pass a pixel independent covariance as an array
        with shape *(n_freq,)*: it has to be *(n_freq, ..., 1)* in order to be
        broadcastable (consider using :func:`basic_comp_sep`, in this case).
        Values equal to `hp.UNSEEN` or, if `MaskedArray`, masked values are
        neglected during the component separation process.
    nside:
        For each pixel of a HEALPix map with this nside, the non-linear
        parameters are estimated independently
    patch_ids: array
        For each pixel, the array stores the id of the region over which to
        perform component separation independently.

    Returns
    -------
    result: dict
	It includes

	- **param**: *(list)* - Names of the parameters fitted
	- **x**: *(ndarray)* - ``x[i]`` is the best-fit (map of) the *i*-th
          parameter
        - **Sigma**: *(ndarray)* - ``Sigma[i, j]`` is the (map of) the
          semi-analytic covariance between the *i*-th and the *j*-th parameter
          It is meaningful only in the high signal-to-noise regime and when the
          *cov* is the true covariance of the data
        - **s**: *(ndarray)* - Component amplitude maps
        - **mask_good**: *(ndarray)* - mask of the entries actually used in the
          component separation

    Note
    ----
    During the component separation, a pixel is masked if at least one of
    its frequencies is masked, either in *data* or in *cov*.

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
                                hp.npix2nside(data.shape[-1]))
        patch_ids_cs = patch_ids[mask]
        res = alg.multi_comp_sep(A_ev, data_cs, invN, A_dB_ev, comp_of_param,
                             patch_ids_cs, x0, **minimize_kwargs)
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
        List storing the :class:`Component` s of the mixing matrix
    instrument
        Instrument object used to define the mixing matrix.
        It can be any object that has what follows either as a key or as an
        attribute (e.g. `dict`, `PySM.Instrument`)

        - **Frequencies**
        - **Sens_I** or **Sens_P** (optional, frequencies are inverse-noise
          weighted according to these noise levels)

    data: ndarray or MaskedArray
        Data vector to be separated. Shape *(n_freq, ..., n_pix).*
        *...* can be

        - absent or 1: temperature maps
        - 2: polarization maps
        - 3: temperature and polarization maps (see note)

        Values equal to `hp.UNSEEN` or, if `MaskedArray`, masked values are
        neglected during the component separation process.
    nside:
        For each pixel of a HEALPix map with this nside, the non-linear
        parameters are estimated independently

    Returns
    -------
    result: dict
	It includes

	- **param**: *(list)* - Names of the parameters fitted
	- **x**: *(ndarray)* - ``x[i]`` is the best-fit (map of) the *i*-th
          parameter
        - **Sigma**: *(ndarray)* - ``Sigma[i, j]`` is the (map of) the
          semi-analytic covariance between the *i*-th and the *j*-th parameter.
          It is meaningful only in the high signal-to-noise regime and when the
          *cov* is the true covariance of the data
        - **s**: *(ndarray)* - Component amplitude maps
        - **mask_good**: *(ndarray)* - mask of the entries actually used in the
          component separation

    Note
    ----

    * During the component separation, a pixel is masked if at least one of
      its frequencies is masked.
    * If you provide temperature and polarization maps, they will constrain the
      **same** set of parameters. In particular, separation is **not** done
      independently for temperature and polarization. If you want an
      independent fitting for temperature and polarization, please launch

      >>> res_T = basic_comp_sep(component_T, instrument, data[:, 0], **kwargs)
      >>> res_P = basic_comp_sep(component_P, instrument, data[:, 1:], **kwargs)

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

    res.mask_good = ~mask
    return res


#Added by Clement Leloup
def harmonic_comp_sep(components, instrument, data, nside, invN=None, mask=None, **minimize_kwargs):
    
    instrument = _force_keys_as_attributes(instrument)
    lmax = 3 * nside - 1
    n_comp = len(components)
    fsky = 1.0
    
    if mask is not None:
        data *= mask
        fsky = float(mask.sum()) / mask.size
    
    print('Computing alms')
    try:
        assert np.any(instrument.Beams)
    except (KeyError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams

    alms = _get_alms(data, beams, lmax)[:,1:,:]
    cl_in = np.array([hp.alm2cl(alm) for alm in alms])
    ell = hp.Alm.getlm(lmax, np.arange(alms.shape[-1]))[0]
    ell = np.stack((ell, ell), axis=-1).reshape(-1) # For transformation into real alms
    #mask_lmin = [l < lmin for l in ell]

    #Add noise to data alms
    #np.random.seed(5)
    nlms_E = [hp.synalm(np.linalg.inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms_B = [hp.synalm(np.linalg.inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms = np.concatenate((np.asarray(nlms_E)[:,np.newaxis,:], np.asarray(nlms_B)[:,np.newaxis,:]), axis=1)
    alms += nlms

    #Produce alms from maps
    alms = _format_alms(alms, lmax)#, mask_lmin)

    #Format the inverse noise matrix
    invNlm = np.array([invN[l,:,:] for l in ell])[:,np.newaxis,:,:]
    #invNlm = invNlm[mask_lmin, ...]

    A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluator(components, instrument)
    if not len(x0):
        A_ev = A_ev()

    # Component separation
    res = alg.comp_sep(A_ev, alms, invNlm, A_dB_ev, comp_of_param, x0, **minimize_kwargs)


    # Craft output
    # 1) Apply the mask, if any
    # 2) Restore the ordering of the input data (pixel dimension last)
    res.params = params
    res.s = np.swapaxes(res.s, 0, 2)
    res.s[res.s == hp.UNSEEN] = 0.
    res.s = np.asarray(res.s, order='C').view(np.complex128)
    cl_out = np.array([hp.alm2cl(alm) for alm in res.s])
    res.cl_in = cl_in/fsky
    res.cl_out = cl_out/fsky
    res.fsky = fsky
    res.chi = res.chi.T
    if 'chi_dB' in res:
        for i in range(len(res.chi_dB)):
            res.chi_dB[i] = res.chi_dB[i].T
    if nside and len(x0):
        res.x = res.x.T
        res.Sigma = res.Sigma.T

    return res
        

def harmonic_ilc(components, instrument, data, lbins=None, weights=None, iter=3):
    """ Internal Linear Combination

    Parameters
    ----------
    components: list or tuple of lists
        `Components` of the mixing matrix. They must have no free parameter.
    instrument: dict or PySM.Instrument
        Instrument object used to define the mixing matrix
        It is required to have:

        - Frequencies

        It may have

        - Beams (FWHM in arcmin) they are deconvolved before ILC

    data: ndarray or MaskedArray
        Data vector to be separated. Shape `(n_freq, ..., n_pix)`. `...` can be
        1, 3 or absent.
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    lbins: array
        It stores the edges of the bins that will have the same ILC weights.
    weights: array
        If provided data are multiplied by the weights map before computing alms

    Returns
    -------
    result : dict
	It includes

        - **W**: *(ndarray)* - ILC weights for each component and possibly each
          patch.
        - **freq_cov**: *(ndarray)* - Empirical covariance for each bin
        - **s**: *(ndarray)* - Component maps
        - **cl_in**: *(ndarray)* - anafast output of the input
        - **cl_out**: *(ndarray)* - anafast output of the output

    Note
    ----

    * During the component separation, a pixel is masked if at least one of its
      frequencies is masked.
    * Output spectra are divided by the fsky. fsky is computed with the MASTER
      formula if `weights` is provided, otherwise it is the fraction of unmasked
      pixels

    """
    instrument = _force_keys_as_attributes(instrument)
    nside = hp.get_nside(data[0])
    lmax = 3 * nside - 1
    lmax = min(lmax, lbins.max())
    n_comp = len(components)
    if weights is not None:
        assert not np.any(_intersect_mask(data) * weights.astype(bool)), \
            "Weights are non-zero where the data is masked"
        fsky = np.mean(weights**2)**2 / np.mean(weights**4)
    else:
        mask = _intersect_mask(data)
        fsky = float(mask.sum()) / mask.size

    print('Computing alms')
    try:
        assert np.any(instrument.Beams)
    except (AttributeError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams

    alms = _get_alms(data, beams, lmax, weights, iter=iter)

    print('Computing ILC')
    res = _harmonic_ilc_alm(components, instrument, alms, lbins, fsky)

    print('Back to real')
    res.s = np.empty((n_comp,) + data.shape[1:], dtype=data.dtype)
    for c in range(n_comp):
        res.s[c] = hp.alm2map(alms[c], nside)

    return res


def _get_alms(data, beams=None, lmax=None, weights=None, iter=3):
    alms = []
    for f, fdata in enumerate(data):
        if weights is None:
            alms.append(hp.map2alm(fdata, lmax=lmax, iter=iter))
        else:
            alms.append(hp.map2alm(hp.ma(fdata)*weights, lmax=lmax, iter=iter))
        print('%i of %i complete' % (f+1, len(data)))
    alms = np.array(alms)

    if beams is not None:
        print('Correcting alms for the beams')
        # FIXME correct polarization with polarization beams
        for fwhm, alm in zip(beams, alms):
            bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax)
            hp.almxfl(alm, 1.0/bl, inplace=True)

    return alms


#Added by Clement Leloup
#Format alms so that they are real and masked
def _format_alms(alms, lmax, mask_lmin=None):

    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    alms[..., np.arange(1, lmax+1, 2)] = hp.UNSEEN  # Mask imaginary m = 0
    mask_alms = _intersect_mask(alms)
    alms[..., mask_alms] = 0  # Thus no contribution to the spectral likelihood
    alms = np.swapaxes(alms, 0, -1)

    if mask_lmin is not None:
        alms[mask_lmin, ...] = 0

    return alms



def _harmonic_ilc_alm(components, instrument, alms, lbins=None, fsky=None):
    cl_in = np.array([hp.alm2cl(alm) for alm in alms])

    # Multipoles for the ILC bins
    lmax = hp.Alm.getlmax(alms.shape[-1])
    ell = hp.Alm.getlm(lmax)[0]
    if lbins is not None:
        ell = np.digitize(ell, lbins)
    # NOTE: use lmax for indexing alms, ell.max() is the maximum bin index

    # Make alms real
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    alms[..., np.arange(1, 2*(lmax+1), 2)] = hp.UNSEEN  # Mask imaginary m = 0
    ell = np.stack((ell, ell), axis=-1).reshape(-1)
    if alms.ndim > 2:  # TEB -> ILC indipendently on each Stokes
        n_stokes = alms.shape[1]
        assert n_stokes in [1, 3], "Alms must be either T only or T E B"
        alms[:, 1:, [0, 2, 2*lmax+2, 2*lmax+3]] = hp.UNSEEN  # EB for ell < 2
        ell = np.stack([ell] * n_stokes)  # Replicate ell for every Stokes
        ell += np.arange(n_stokes).reshape(-1, 1) * (ell.max() + 1) # Add offset

    res = ilc(components, instrument, alms, ell)

    # Craft output
    res.s[res.s == hp.UNSEEN] = 0.
    res.s = np.asarray(res.s, order='C').view(np.complex128)
    cl_out = np.array([hp.alm2cl(alm) for alm in res.s])

    res.cl_in = cl_in
    res.cl_out = cl_out
    if fsky:
        res.cl_in /= fsky
        res.cl_out /= fsky

    res.fsky = fsky
    lrange = np.arange(lmax+1)
    ldigitized = np.digitize(lrange, lbins)
    res.l_ref = (np.bincount(ldigitized, lrange * 2*lrange+1)
                 / np.bincount(ldigitized, 2*lrange+1))
    res.freq_cov *= 2  # sqrt(2) missing between complex-real alm conversion
    if res.s.ndim > 2:
        res.freq_cov.reshape(n_stokes, -1, *res.freq_cov.shape[1:])

    return res


def ilc(components, instrument, data, patch_ids=None):
    """ Internal Linear Combination

    Parameters
    ----------
    components: list or tuple of lists
        `Components` of the mixing matrix. They must have no free parameter.
    instrument: PySM.Instrument
        Instrument object used to define the mixing matrix
        It is required to have:

        - Frequencies

        It's only role is to evaluate the `components` at the
        `instrument.Frequencies`.
    data: ndarray or MaskedArray
        Data vector to be separated. Shape `(n_freq, ..., n_pix)`. `...` can be
        also absent.
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    patch_ids: array
        It stores the id of the region over which the ILC weights are computed
        independently. It must be broadcast-compatible with data.

    Returns
    -------
    result : dict
	It includes

        - **W**: *(ndarray)* - ILC weights for each component and possibly each
          patch.
        - **freq_cov**: *(ndarray)* - Empirical covariance for each patch
        - **s**: *(ndarray)* - Component maps

    Note
    ----
    * During the component separation, a pixel is masked if at least one of its
      frequencies is masked.
    """
    # Checks
    instrument = _force_keys_as_attributes(instrument)
    np.broadcast(data, patch_ids)
    n_freq = data.shape[0]
    assert len(instrument.Frequencies) == n_freq,\
        "The number of frequencies does not match the number of maps provided"
    n_comp = len(components)

    # Prepare mask and set to zero all the frequencies in the masked pixels:
    # NOTE: mask are good pixels
    mask = ~_intersect_mask(data)

    mm = MixingMatrix(*components)
    A = mm.eval(instrument.Frequencies)

    data = data.T
    res = OptimizeResult()
    res.s = np.full(data.shape[:-1] + (n_comp,), hp.UNSEEN)

    def ilc_patch(mask_pix, mask_id):
        if not np.any(mask_pix):
            return
        data_patch = data[mask_pix]  # data_patch is a copy (advanced indexing)
        res.freq_cov[mask_id] = np.cov(data_patch.reshape(-1, n_freq).T)
        assert np.linalg.cond(res.freq_cov[mask_id]) < 1e8,\
            "Empirical covariance matrix cannot be reliably inverted"
        inv_freq_cov = np.linalg.inv(res.freq_cov[mask_id])
        res.W[mask_id] = alg.W(A, inv_freq_cov)
        res.s[mask_pix] = alg._mv(res.W[mask_id], data_patch)

    if patch_ids is None:
        res.freq_cov = np.full((n_freq, n_freq), hp.UNSEEN)
        res.W = np.full((n_comp, n_freq), hp.UNSEEN)
        ilc_patch(mask, np.s_[:])
    else:
        n_id = patch_ids.max() + 1
        res.freq_cov = np.full((n_id, n_freq, n_freq), hp.UNSEEN)
        res.W = np.full((n_id, n_comp, n_freq), hp.UNSEEN)
        for i in range(n_id):
            mask_i = ((patch_ids == i) & mask).T
            ilc_patch(mask_i, i)

    res.s = res.s.T
    res.components = mm.components

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


# What are _LowerCaseAttrDict and _force_keys_as_attributes?
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
