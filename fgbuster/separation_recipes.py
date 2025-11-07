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
import logging
from glob import glob
import os.path as op
import os
import numpy as np
from scipy.optimize import OptimizeResult
import healpy as hp
from . import algebra as alg
from .mixingmatrix import MixingMatrix
from .observation_helpers import standardize_instrument


__all__ = [
    'basic_comp_sep',
    'weighted_comp_sep',
    'ilc',
    'harmonic_ilc',
    'harmonic_ilc_alm',
    'adaptive_comp_sep',
    'multi_res_comp_sep',
]


def weighted_comp_sep(components, instrument, data, cov, nside=0,
                      **minimize_kwargs):
    """ Weighted component separation

    Parameters
    ----------
    components: list or tuple of lists
        List storing the :class:`Component` s of the mixing matrix
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency**

        It can be anything that is convertible to a float numpy array.
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
    instrument = standardize_instrument(instrument)
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
    if len(x0) == 0:
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
        missing_ids = hp.nside2npix(nside) - par_array.shape[0]
        extra_dims = np.full((missing_ids,) + par_array.shape[1:], hp.UNSEEN)
        result = np.concatenate((par_array, extra_dims))
        result[np.isnan(result)] = hp.UNSEEN
        return result.T

    if len(x0) > 0:
        if 'chi_dB' in res:
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
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency**
        - **depth_i** or **depth_p** (optional, frequencies are inverse-noise
          weighted according to these noise levels)

        They can be anything that is convertible to a float numpy array.
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
    instrument = standardize_instrument(instrument)
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
    if len(x0) == 0:
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
    if nside and len(x0) > 0:
        x_mask = hp.ud_grade(mask.astype(float), nside) == 1.
        res.x[x_mask] = hp.UNSEEN
        res.Sigma[x_mask] = hp.UNSEEN
        res.x = res.x.T
        res.Sigma = res.Sigma.T

    res.mask_good = ~mask
    return res


def adaptive_comp_sep(components, instrument, data, patch_ids,
                      **minimize_kwargs):
    """ Arbitrary clusters for each parameter

    Parameters
    ----------
    components: list
        List storing the :class:`Component` s of the mixing matrix
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency**
        - **depth_i** or **depth_p** (optional, frequencies are inverse-noise
          weighted according to these noise levels)

        They can be anything that is convertible to a float numpy array.
    data: ndarray or MaskedArray
        Data vector to be separated. Shape *(n_freq, ..., n_pix).*
        *...* can be

        - absent or 1: temperature maps
        - 2: polarization maps
        - 3: temperature and polarization maps (see note)

        Values equal to `hp.UNSEEN` or, if `MaskedArray`, masked values are
        neglected during the component separation process.
    patch_ids: list
        The *i*-th element is the clusters map of the *i*-th parameter.
        A cluster map is a map of integers that, for each pixel defines the
        index of the cluster the pixel belongs to.
    minimize_kwargs: dict
        kwargs of `scipy.optimize.minimize`. In addition it allows for
        saving/restoring checkpoints. Add the following dictionary to
        `minimize_kwargs['checkpoint']`::

            # The values are the defaults
            {
                'odir': './',
                # Save iteraton x to `odir/iter_x.npy`
                'start': 0,
                # Start from this iteration, If not provided use that largest
                # stored iteration
                'delta': 1,
                # Save a checkpoint every `delta` iterations
            }

    Returns
    -------
    result: dict
	It includes

	- **param**: *(list)* - Names of the parameters fitted
	- **x**: *(seq)* - ``x[i][j]`` is the best-fit values of the *j*-th
          clusters of the *i*-th parameter.
	- **x_map**: *(seq)* - ``x[i]`` is the map of the *i*-th parameter.
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
    instrument = standardize_instrument(instrument)

    # Prepare mask and set to zero all the frequencies in the masked pixels:
    # NOTE: mask are bad pixels
    mask = _intersect_mask(data)
    data = hp.pixelfunc.ma_to_array(data).copy()
    data[..., mask] = 0  # Thus no contribution to the spectral likelihood

    try:
        data_nside = hp.get_nside(data[0])
    except TypeError:
        raise ValueError("data has to be a stack of healpix maps")

    prewhiten_factors = _get_prewhiten_factors(instrument, data.shape, data_nside)
    invN = np.zeros(prewhiten_factors.shape+prewhiten_factors.shape[-1:])
    np.einsum('...ii->...i', invN)[:] = prewhiten_factors**2

    for ids in patch_ids:
        assert np.all(ids >= 0)
        assert ids.dtype.kind in 'ui'
    n_clusters = [ids.max()+1 for ids in patch_ids]

    def array2maps(x):
        i = 0
        maps = []
        for n_cluster, ids in zip(n_clusters, patch_ids):
            maps.append(x[i:i+n_cluster][ids])
            i += n_cluster
        return maps

    extra_dim = [1] * (data.ndim - 2)
    unpack = lambda x: [m.reshape(-1, *extra_dim) for m in array2maps(x)]

    try:
        checkpoint_dir = minimize_kwargs['checkpoint'].get('odir', './')
        os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            x0 = np.load(op.join(checkpoint_dir, f"iter_{minimize_kwargs['checkpoint']['start']}.npy"))
            logging.warning(f"Iteration number {minimize_kwargs['checkpoint']['start']} loaded")
        except (OSError, KeyError):  # Either start is not set, or the file is missing
            iter_files = glob(op.join(checkpoint_dir, 'iter_*.npy'))
            iter_ids = [int(op.splitext(op.basename(f))[0].split('_')[1])
                        for f in iter_files]
            i_iter = max(iter_ids+[0])
            logging.warning(f'Highest iteration number found is {i_iter}')
            x0 = np.load(iter_files[iter_ids.index(i_iter)])
            minimize_kwargs['checkpoint']['start'] = i_iter
        if 'options' in minimize_kwargs and 'maxiter' in minimize_kwargs['options']:
            minimize_kwargs['options']['maxiter'] -= i_iter
    except (KeyError, ValueError):
        x0 = [x for c in components for x in c.defaults]
        x0 = [np.full(n_cluster, px0) for n_cluster, px0 in zip(n_clusters, x0)]
        x0 = np.concatenate(x0)

    A = MixingMatrix(*components)
    assert A.n_param == len(patch_ids), (
        "%i free parameters but %i patch_ids"
        % (len(A.defaults), len(patch_ids)))
    A_ev = A.evaluator(instrument.frequency, unpack)
    A_dB_ev = A.diff_evaluator(instrument.frequency, unpack)
    end_w_last_checkpoint = (
        'options' in minimize_kwargs
        and 'maxiter' in minimize_kwargs['options']
        and minimize_kwargs['options']['maxiter'] < 1
    )
    if end_w_last_checkpoint:
        A_ev = A_ev(x0)

    comp_of_dB = list(zip(A.comp_of_dB, patch_ids))
    bounds = minimize_kwargs.get('bounds')
    if bounds is not None:
        minimize_kwargs['bounds'] = _get_bounds(patch_ids, bounds)

    # Component separation
    res = alg.comp_sep(A_ev, data.T, invN, A_dB_ev, comp_of_dB, x0,
                       **minimize_kwargs)

    if end_w_last_checkpoint:
        res.x = x0
    # Craft output
    # 1) Apply the mask, if any
    # 2) Restore the ordering of the input data (pixel dimension last)
    def mask_transpose(x):
        x[mask] = hp.UNSEEN
        return x.T

    res.params = A.params
    res.s = mask_transpose(res.s)
    res.chi = mask_transpose(res.chi)
    res.x_map = array2maps(res.x)
    for m in res.x_map:
        m[mask] = hp.UNSEEN

    res.x = [res.x[stop-n:stop]
             for n, stop in zip(n_clusters, np.cumsum(n_clusters))]
    for x, ids, n_cluster in zip(res.x, patch_ids, n_clusters):
        # Clusters witn no valid pixels are set to UNSEEN
        x[np.bincount(ids[~mask], minlength=n_cluster) == 0] = hp.UNSEEN

    if 'chi_dB' in res:
        for i in range(len(res.chi_dB)):
            res.chi_dB[i] = mask_transpose(res.chi_dB[i])

    res.mask_good = ~mask
    return res


def multi_res_comp_sep(components, instrument, data, nsides, **minimize_kwargs):
    """ Basic component separation

    Parameters
    ----------
    components: list
        List storing the :class:`Component` s of the mixing matrix
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency**
        - **depth_i** or **depth_p** (optional, frequencies are inverse-noise
          weighted according to these noise levels)

        They can be anything that is convertible to a float numpy array.
    data: ndarray or MaskedArray
        Data vector to be separated. Shape *(n_freq, ..., n_pix).*
        *...* can be

        - absent or 1: temperature maps
        - 2: polarization maps
        - 3: temperature and polarization maps (see note)

        Values equal to `hp.UNSEEN` or, if `MaskedArray`, masked values are
        neglected during the component separation process.
    nsides: seq
        Specify the ``nside`` for each free parameter of the components

    Returns
    -------
    result: dict
	See `adaptive_comp_sep`

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
    nside_data = hp.get_nside(data[0])
    patch_ids = [
        _my_ud_grade(np.arange(_my_nside2npix(nside)), nside_data).astype(int)
        for nside in nsides]
    return adaptive_comp_sep(components, instrument, data, patch_ids,
                             **minimize_kwargs)


def harmonic_ilc(components, instrument, data, lbins=None, weights=None, iter=3):
    """ Harmonic Internal Linear Combination

    Parameters
    ----------
    components: list or tuple of lists
        `Components` of the mixing matrix. They must have no free parameter.
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency**
        - **fwhm** (arcmin) they are deconvolved before ILC

        They can be anything that is convertible to a float numpy array.
    data: ndarray or MaskedArray
        Data vector to be separated. Shape ``(n_freq, ..., n_pix)``.
        ``...`` can be 1, 3 or absent. If 3, the separation is done independently
	fot T, E and B.
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    lbins: array
        It stores the edges of the bins that will have the same ILC weights.
        If a multipole is not in a bin but is the alms, an independent bin
	will be assigned to it
    weights: array
        If provided data are multiplied by the weights map before computing alms

    Returns
    -------
    result : dict
	It includes

        - **W**: *(ndarray)* - ILC weights for each component and possibly
	  each index of the `...` dimension in the alms.
        - **s**: *(ndarray)* - Component maps
        - **cl_in**: *(ndarray)* - Spectra of the input alm
        - **cl_out**: *(ndarray)* - Spectra of the output alm
        - **fsky**: *(ndarray)* - The input fsky used to correct the cls

    Note
    ----

    * During the component separation, a pixel is masked if at least one of its
      frequencies is masked.
    * Output spectra are divided by the fsky. fsky is computed with the MASTER
      formula if `weights` is provided, otherwise it is the fraction of unmasked
      pixels

    """
    instrument = standardize_instrument(instrument)
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

    logging.info('Computing alms')
    try:
        assert np.any(instrument.fwhm)
    except (AttributeError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.fwhm

    alms = _get_alms(data, beams, lmax, weights, iter=iter)

    logging.info('Computing ILC')
    res = harmonic_ilc_alm(components, instrument, alms, lbins, fsky)

    logging.info('Back to real')
    alms = res.s
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
        logging.info(f"{f+1} of {len(data)} complete")
    alms = np.array(alms)

    if beams is not None:
        logging.info('Correcting alms for the beams')
        for fwhm, alm in zip(beams, alms):
            bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax, pol=(alm.ndim==2))
            if alm.ndim == 1:
                alm = [alm]
                bl = [bl]

            for i_alm, i_bl in zip(alm, bl.T):
                hp.almxfl(i_alm, 1.0/i_bl, inplace=True)

    return alms


def _apply_harmonic_W(W,  # (..., ell, comp, freq)
                      alms):  # (freq, ..., lm)
    lmax = hp.Alm.getlmax(alms.shape[-1])
    res = np.full((W.shape[-2],) + alms.shape[1:], np.nan, dtype=alms.dtype)
    start = 0
    for i in range(0, lmax+1):
        n_m = lmax + 1 - i
        res[..., start:start+n_m] = np.einsum('...lcf,f...l->c...l',
                                              W[..., i:, :, :],
                                              alms[..., start:start+n_m])
        start += n_m
    return res


def harmonic_ilc_alm(components, instrument, alms, lbins=None, fsky=None):
    """ Internal Linear Combination of alms

    Parameters
    ----------
    components: list or tuple of lists
        `Components` of the mixing matrix. They must have no free parameter.
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency**

        It can be anything that is convertible to a float numpy array.
    alms: ndarray
        Data vector to be separated. Shape ``(n_freq, ..., lm)``.
        ``...`` can be 1, 3 or absent. The ILC weights are computed
	independently for each of its indices.
    lbins: array
        It stores the edges of the bins that will have the same ILC weights.
	If a multipole is not in a bin but is the alms, an independent bin
	will be assigned to it
    fsky: array
        If provided the output power spectra are corrected for this factor

    Returns
    -------
    result : dict
	It includes

        - **W**: *(ndarray)* - ILC weights for each component and possibly
	  each index of the `...` dimension in the alms.
        - **s**: *(ndarray)* - Alms of the cleaned components
        - **cl_in**: *(ndarray)* - Spectra of the input alm
        - **cl_out**: *(ndarray)* - Spectra of the output alm
        - **fsky**: *(ndarray)* - The input fsky used to correct the cls

    """
    cl_in = np.array([hp.alm2cl(alm) for alm in alms])

    mm = MixingMatrix(*components)
    A = mm.eval(instrument.frequency)

    cov = _empirical_harmonic_covariance(alms)
    if lbins is not None:
        for lmin, lmax in zip(lbins[:-1], lbins[1:]):
            # Average the covariances in the bin
            lmax = min(lmax, cov.shape[-1])
            dof = 2 * np.arange(lmin, lmax) + 1
            cov[..., lmin:lmax] = (
                (dof / dof.sum() * cov[..., lmin:lmax]).sum(-1)
                )[..., np.newaxis]
    cov = _regularized_inverse(cov.swapaxes(-1, -3))
    ilc_filter = np.linalg.inv(A.T @ cov @ A) @ A.T @ cov
    del cov, dof

    res = OptimizeResult()
    res.s = _apply_harmonic_W(ilc_filter, alms)

    # Craft output
    cl_out = np.array([hp.alm2cl(alm) for alm in res.s])
    res.cl_in = cl_in
    res.cl_out = cl_out
    if fsky:
        res.cl_in /= fsky
        res.cl_out /= fsky

    res.fsky = fsky
    res.W = ilc_filter

    return res


def _empirical_harmonic_covariance(alms):
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64).reshape(alms.shape+(2,))
    if alms.ndim > 3:  # Shape has to be ([Stokes], freq, lm, ri)
        alms = alms.transpose(1, 0, 2, 3)
    lmax = hp.Alm.getlmax(alms.shape[-2])

    res = (alms[..., np.newaxis, :, :lmax+1, 0]
           * alms[..., :, np.newaxis, :lmax+1, 0])  # (Stokes, freq, freq, ell)


    consumed = lmax + 1
    for i in range(1, lmax+1):
        n_m = lmax + 1 - i
        alms_m = alms[..., consumed:consumed+n_m, :]
        res[..., i:] += 2 * np.einsum('...fli,...nli->...fnl', alms_m, alms_m)
        consumed += n_m

    res /= 2 * np.arange(lmax + 1) + 1
    return res


def _regularized_inverse(cov):
    """ Covariance pseudo-inverse

    Regularize cov with the diagonal (i.e. invert the correlation matrix).
    If a row/col is noise-dominated and the noise is mostly diagonal, this
    regularization prevents the signal from being lost in the pseudo-inverse.

    Infinity and NaN are set to zero, thus overflows due to noise explosions
    (e.g. due to beam corrections) are properly handled
    """
    inv_std = np.einsum('...ii->...i', cov)
    inv_std = 1 / np.sqrt(inv_std)
    np.nan_to_num(inv_std, False, 0, 0, 0)
    np.nan_to_num(cov, False, 0, 0, 0)

    inv_cov = np.linalg.pinv(cov
                             * inv_std[..., np.newaxis]
                             * inv_std[..., np.newaxis, :])
    return inv_cov * inv_std[..., np.newaxis] * inv_std[..., np.newaxis, :]


def ilc(components, instrument, data, patch_ids=None):
    """ Internal Linear Combination

    Parameters
    ----------
    components: list or tuple of lists
        `Components` of the mixing matrix. They must have no free parameter.
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency**

        They can be anything that is convertible to a float numpy array.
    data: ndarray or MaskedArray
        Data vector to be separated. Shape ``(n_freq, ..., n_pix)``.
        ``...`` can be also absent.
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
    instrument = standardize_instrument(instrument)
    np.broadcast(data, patch_ids)
    n_freq = data.shape[0]
    assert len(instrument.frequency) == n_freq,\
        "The number of frequencies does not match the number of maps provided"
    n_comp = len(components)

    # Prepare mask and set to zero all the frequencies in the masked pixels:
    # NOTE: mask are good pixels
    mask = ~_intersect_mask(data)

    mm = MixingMatrix(*components)
    A = mm.eval(instrument.frequency)

    data = data.T
    res = OptimizeResult()
    res.s = np.full(data.shape[:-1] + (n_comp,), hp.UNSEEN)

    def ilc_patch(ids_i, i_patch):
        if not np.any(ids_i):
            return
        data_patch = data[ids_i]  # data_patch is a copy (advanced indexing)
        cov = np.cov(data_patch.reshape(-1, n_freq).T)
        # Perform the inversion of the correlation instead of the covariance.
        # This allows to meaninfully invert covariances that have very noisy
        # channels.
        assert cov.ndim == 2
        cov_regularizer = np.diag(cov)**0.5 * np.diag(cov)[:, np.newaxis]**0.5
        correlation = cov / cov_regularizer
        try:
            inv_freq_cov = np.linalg.inv(correlation) / cov_regularizer
        except np.linalg.LinAlgError:
            np.set_printoptions(precision=2)
            logging.error(
                f"Empirical covariance matrix cannot be reliably inverted.\n"
                f"The domain that failed is {i_patch}.\n"
                f"Covariance matrix diagonal {np.diag(cov)}\n"
                f"Correlation matrix\n{correlation}")
            raise
        res.freq_cov[i_patch] = cov
        res.W[i_patch] = alg.W(A, inv_freq_cov)
        res.s[ids_i] = alg._mv(res.W[i_patch], data_patch)

    if patch_ids is None:
        res.freq_cov = np.full((n_freq, n_freq), hp.UNSEEN)
        res.W = np.full((n_comp, n_freq), hp.UNSEEN)
        ilc_patch(mask, np.s_[:])
    else:
        n_id = patch_ids.max() + 1
        res.freq_cov = np.full((n_id, n_freq, n_freq), hp.UNSEEN)
        res.W = np.full((n_id, n_comp, n_freq), hp.UNSEEN)
        patch_ids_bak = patch_ids.copy().T
        patch_ids_bak[~mask] = -1
        for i in range(n_id):
            ids_i = np.where(patch_ids_bak == i)
            ilc_patch(ids_i, i)

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
            sens = instrument.depth_i
        elif data_shape[1] == 2:
            sens = instrument.depth_p
        elif data_shape[1] == 3:
            sens = np.stack(
                (instrument.depth_i, instrument.depth_p, instrument.depth_p))
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
    A_ev = A.evaluator(instrument.frequency)
    A_dB_ev = A.diff_evaluator(instrument.frequency)
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


def _get_bounds(idss, bounds):
    res = []
    for ids, bound in zip(idss, bounds):
        n_clusters = ids.max(-1) + 1
        res += [bound] * n_clusters
    return res


def _my_nside2npix(nside):
    if nside:
        return hp.nside2npix(nside)
    else:
        return 1


def _my_ud_grade(map_in, nside_out, **kwargs):
    # As healpy.ud_grade, but it accepts map_in of nside = 0 and nside_out = 0,
    # which in this module means a single float or lenght-1 array
    if nside_out == 0:
        try:
            # Both input and output have nside = 0
            return np.array([float(map_in)])
        except TypeError:
            # This is really clunky...
            # 1) Downgrade to nside 1
            # 2) put the 12 values in the pixels of a nside 4 map that belong to
            #    the same nside 1 pixels
            # 3) Downgrade to nside 1
            # 4) pick the value of the pixel in which the 12 values were placed
            map_in = hp.ud_grade(map_in, 1, **kwargs)
            out = np.full(hp.nside2npix(4), hp.UNSEEN)
            ids = hp.ud_grade(np.arange(12), 4, **kwargs)
            out[np.where(ids == 0)[0][:12]] = map_in
            kwargs['pess'] = False
            res = hp.ud_grade(out, 1, **kwargs)
            return res[:1]
    try:
        # Input has nside = 0 (or 1)
        return hp.ud_grade(np.ones(12) * map_in,
                           nside_out, **kwargs)
    except ValueError:
        # Fall back to standard healpy ud_grade
        return hp.ud_grade(map_in, nside_out, **kwargs)


def _intersect_mask(maps):
    if hp.pixelfunc.is_ma(maps):
        mask = maps.mask
    else:
        mask = maps == hp.UNSEEN

    # Mask entire pixel if any of the frequencies in the pixel is masked
    return np.any(mask, axis=tuple(range(maps.ndim-1)))
