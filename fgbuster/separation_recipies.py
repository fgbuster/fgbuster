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
import scipy as sp
from scipy.optimize import OptimizeResult
import healpy as hp
import algebra as alg
from .mixingmatrix import MixingMatrix
from .component_model import CMB, SemiBlind
import sys
import numdifftools

np.set_printoptions(threshold=sys.maxsize)

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
    #A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluator(
    #    components, instrument, prewhiten_factors=prewhiten_factors)
    A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluator(
        components, instrument)
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
def harmonic_semiblind(components, instrument, templates, data, nside, invN=None, **minimize_kwargs):
    """ Semi-blind method

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

    templates: str
        Name of templates file.
    data: ndarray or MaskedArray
        Data vector to be separated. Shape `(n_freq, ..., n_pix)`. `...` can be
        1, 3 or absent.
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    invN: ndarray
        Inverse noise matrix

    Returns
    -------
    result : dict
	It includes

        - **s**: *(ndarray)* - Component maps

    Note
    ----

    * During the component separation, a pixel is masked if at least one of its
      frequencies is masked.
    * Works just with polarization at the moment

    """
    instrument = _force_keys_as_attributes(instrument)
    lmax = 3 * nside - 1
    n_comp = len(components)
    #mask = _intersect_mask(data)
    #fsky = float(mask.sum()) / mask.size
    #data[..., mask] = 0
    mask = hp.read_map("fgbuster/templates/HFI_Mask_GalPlane-apo2_2048_R2.00.fits", field=(2))
    mask = hp.ud_grade(mask, nside_out=nside)
    data *= mask
    
    print('Computing alms')
    try:
        assert np.any(instrument.Beams)
    except (KeyError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams

    alms = _get_alms(data, beams, lmax)[:,1:,:]
    ell = hp.Alm.getlm(lmax, np.arange(alms.shape[-1]))[0]

    #Add noise to data alms
    nlms_E = [hp.synalm(alg._inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms_B = [hp.synalm(alg._inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms = np.concatenate((np.asarray(nlms_E)[:,np.newaxis,:], np.asarray(nlms_B)[:,np.newaxis,:]), axis=1)
    alms += nlms

    #Produce alms from maps
    lmin = 30
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    alms[..., np.arange(1, lmax+1, 2)] = hp.UNSEEN  # Mask imaginary m = 0
    mask_alms = _intersect_mask(alms)
    alms[..., mask_alms] = 0  # Thus no contribution to the spectral likelihood
    alms = np.swapaxes(alms, 0, 2)
    ell = np.stack((ell, ell), axis=-1).reshape(-1)
    mask_lmin = [l < lmin for l in ell]
    alms[mask_lmin, ...] = 0
    print('alms : ', alms.shape)


    print('Computing prior')
    if isinstance(templates, np.ndarray):
        ell_in = np.arange(2, lmax+1)
        cl_in = np.concatenate((np.zeros((2, 2)), 2*np.pi*templates[1:3,:lmax-1]/(ell_in*(ell_in+1))), axis=1) #Take only polarization hard coded here
    else:
        cl_in = hp.read_cl(templates)[1:3,:lmax+1] #Take only polarization
        
    #Format the prior shape
    with np.errstate(divide='ignore'):
        EE_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[0,:]]) #Should modify here in case several non-blind components
        BB_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[1,:]])
        cl_in = np.stack((EE_in, BB_in), axis=1) #Probably a better way to do that
    cl_in[~np.isfinite(cl_in)] = 0.
    prior = np.array([cl_in[l,:,:] for l in ell])#hp.Alm.getlm(lmax, np.arange(alms.shape[0]))[0]])
    #prior = None

    
    #Format the inverse noise matrix
    invNlm = np.array([invN[l,:,:] for l in ell])[:,np.newaxis,:,:]#hp.Alm.getlm(lmax, np.arange(alms.shape[0]))[0]])[:,np.newaxis,:,:]

    print('Computing mixing matrix')
    #A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluator(components, instrument)
    A_ev, A_dB_ev, comp_of_param, x0, params = init_semiblind_mixmat(components, instrument)
    if not len(x0):
        A_ev = A_ev()
  
    #print('Separating components')
    #res = alg.semiblind_comp_sep(A_ev(x0), alms, invNlm, prior, A_dB_ev, comp_of_param, x0, **minimize_kwargs)

    x, y, res = grid_test(x0, 50, 0.003, 0.003, A_ev, alms, prior, invNlm)

    x = np.asarray(x)
    y = np.asarray(y)
    res = np.asarray(res)
    

    #return cl_in, invN
    return x, y, res


#Added by Clement Leloup
def semiblind(components, instrument, templates, data, nside, invN=None, **minimize_kwargs):
    """ Semi-blind method

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

    templates: str
        Name of templates file.
    data: ndarray or MaskedArray
        Data vector to be separated. Shape `(n_freq, ..., n_pix)`. `...` can be
        1, 3 or absent.
        Values equal to hp.UNSEEN or, if MaskedArray, masked values are
        neglected during the component separation process.
    invN: ndarray
        Inverse noise matrix

    Returns
    -------
    result : dict
	It includes

        - **s**: *(ndarray)* - Component maps

    Note
    ----

    * During the component separation, a pixel is masked if at least one of its
      frequencies is masked.
    * Works just with polarization at the moment

    """
    instrument = _force_keys_as_attributes(instrument)
    lmax = 3 * nside - 1
    n_comp = len(components)
    #mask = _intersect_mask(data)
    #data[..., mask] = 0
    #mask = hp.read_map("fgbuster/templates/HFI_Mask_GalPlane-apo2_2048_R2.00.fits", field=(2))
    #mask = hp.ud_grade(mask, nside_out=nside)
    #data *= mask
    #fsky = float(mask.sum()) / mask.size
    #print('fsky = {}%'.format(fsky*100))

    try:
        assert np.any(instrument.Beams)
    except (KeyError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams

    print('Computing mixing matrix')
    #A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluator(components, instrument)
    A_ev, A_dB_ev, comp_of_param, x0, params = init_semiblind_mixmat(components, instrument)
    if not len(x0):
        A_ev = A_ev()
  
    print('Computing prior')
    data = np.swapaxes(data, 0, 2)
    #prior = np.zeros((data.shape[0], data.shape[1], A_ev(x0).shape[1], A_ev(x0).shape[1]))
    prior = None
    #print('prior : ', prior.shape)
    print('data : ', data.shape)

    #print('Separating components')
    #res = alg.semiblind_comp_sep(A_ev, alms, invNlm, prior, A_dB_ev, comp_of_param, x0, **minimize_kwargs)

    '''
    print('Testing likelihood computation')
    #prior = np.zeros(prior.shape)
    x = [x0 for i in np.arange(40)]
    x = np.asarray(x)
    print(x0)
    for i in np.arange(x.shape[0]):
        x[i,2] = x0[2]+0.49*i-10
    res = [alg.semiblind_logL(A_ev(x[i,:]), alms, prior, invNlm) for i in np.arange(x.shape[0])]
    res2 = [alg.semiblind_logL_bruteforce(A_ev(x[i,:]), alms, prior, invNlm) for i in np.arange(x.shape[0])]
    res3 = [alg.logL(A_ev(x[i,:]), alms,invNlm) for i in np.arange(x.shape[0])]
    #res = [alg.semiblind_logL_dB(A_ev(x[i,:]), alms, prior, invNlm, A_dB_ev((x[i,:])), comp_of_param) for i in np.arange(x.shape[0])]
    #res2 = [alg.semiblind_logL_dB_bruteforce(A_ev(x[i,:]), alms, prior, invNlm, A_dB_ev((x[i,:])), comp_of_param) for i in np.arange(x.shape[0])]
    #res3 = [alg.logL_dB(A_ev(x[i,:]), alms, invNlm, A_dB_ev((x[i,:])), comp_of_param) for i in np.arange(x.shape[0])]

    res = np.asarray(res)
    res2 = np.asarray(res2)
    res3 = np.asarray(res3)
    
    #Craft output
    #Empty atm
    '''

    #prior = np.zeros(prior.shape)
    print('x0 = ', x0)
    x, y, res = grid_test(x0, 20, 10, 10, A_ev, data, prior, invN)

    x = np.asarray(x)
    y = np.asarray(y)
    res = np.asarray(res)

    #return x[:,2], res, res2, res3
    #return x[:,2], res[:,2], res2[:,2], res3[:,2]

    return x, y, res


#Added by Clement Leloup
def grid_test(x0, n_pts, gap1, gap2, A_ev, data, prior, invN):

    #print(x0)
    x0 = np.array([0.458394863178, 1.0, 2.34401664336])
    
    x1 = [x0[0]-gap1+2*gap1*i/n_pts for i in np.arange(n_pts)]
    y1 = [x0[-1]-gap2+2*gap2*i/n_pts for i in np.arange(n_pts)]
    xy = np.meshgrid(x1, y1)
    x_flat = xy[0].flatten()
    y_flat = xy[1].flatten()

    x = [x0 for i in np.arange(n_pts**2)]
    x = np.asarray(x)
    #for i in np.arange(n_pts**2):
    x[:,0] = np.asarray(x_flat)
    x[:,-1] = np.asarray(y_flat)

    res = [alg.semiblind_logL(A_ev(x[i,:]), data, prior, invN) for i in np.arange(x.shape[0])]

    return x[:, 0], x[:, -1], res


#Added by Clement Leloup
def test_fisher(components, instrument, templates, data, nside, nblind, invN=None, **minimize_kwargs):

    instrument = _force_keys_as_attributes(instrument)
    lmax = 3 * nside - 1
    n_comp = len(components)
    mask = hp.read_map("fgbuster/templates/HFI_Mask_GalPlane-apo2_2048_R2.00.fits", field=(2))
    mask = hp.ud_grade(mask, nside_out=nside)
    #data *= mask
    #fsky = float(mask.sum()) / mask.size
    fsky = 1
    
    print('Computing alms')
    try:
        assert np.any(instrument.Beams)
    except (KeyError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams

    alms = _get_alms(data, beams, lmax)[:,1:,:]
    ell = hp.Alm.getlm(lmax, np.arange(alms.shape[-1]))[0]

    #Add noise to data alms
    nlms_E = [hp.synalm(alg._inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms_B = [hp.synalm(alg._inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms = np.concatenate((np.asarray(nlms_E)[:,np.newaxis,:], np.asarray(nlms_B)[:,np.newaxis,:]), axis=1)
    alms += nlms

    #Produce alms from maps
    lmin = 30
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    alms[..., np.arange(1, lmax+1, 2)] = hp.UNSEEN  # Mask imaginary m = 0
    mask_alms = _intersect_mask(alms)
    alms[..., mask_alms] = 0  # Thus no contribution to the spectral likelihood
    alms = np.swapaxes(alms, 0, 2)
    ell = np.stack((ell, ell), axis=-1).reshape(-1)
    mask_lmin = [l < lmin for l in ell]
    alms[mask_lmin, ...] = 0


    print('Computing prior')
    if isinstance(templates, np.ndarray):
        ell_in = np.arange(2, lmax+1)
        cl_in = np.concatenate((np.zeros((2, 2)), 2*np.pi*templates[1:3,:lmax-1]/(ell_in*(ell_in+1))), axis=1) #Take only polarization hard coded here
    else:
        cl_in = hp.read_cl(templates)[1:3,:lmax+1] #Take only polarization
        
    #Format the prior shape
    with np.errstate(divide='ignore'):
        EE_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[0,:]]) #Should modify here in case several non-blind components
        BB_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[1,:]])
        cl_in = np.stack((EE_in, BB_in), axis=1) #Probably a better way to do that
    cl_in[~np.isfinite(cl_in)] = 0.
    prior = np.array([cl_in[l,:,:] for l in ell])/fsky
    #prior = None

    
    #Format the inverse noise matrix
    invNlm = np.array([invN[l,:,:] for l in ell])[:,np.newaxis,:,:]

    print('Computing mixing matrix')
    A_ev, A_dB_ev, comp_of_param, x0, params = init_semiblind_mixmat(components, instrument, nblind)
    if not len(x0):
        A_ev = A_ev()

    print(x0)
        
    #Maximize likelihood
    A_dB_ev, comp_of_dB = alg._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)
    fun, jac, last_values = alg._semiblind_build_bound_inv_logL_and_logL_dB(
        A_ev, alms, invNlm, prior, A_dB_ev, comp_of_dB) #modification here
    minimize_kwargs['jac'] = jac
    res = sp.optimize.minimize(fun, x0, **minimize_kwargs)

    # Gather results
    u_e_v_last, A_dB_last, x_last, pw_d = last_values
    if not np.all(x_last[0] == res.x):
        fun(res.x) #  Make sure that last_values refer to the minimum
    res.s = alg._semiblind_Wd_svd(u_e_v_last[0], prior, pw_d[0])

    #prior = np.zeros(prior.shape)
    #L_brute = numdifftools.Hessian(fun)(res.x)
    #L_param = alg._fisher_logL_dB_dB_svd(u_e_v_last[0], res.s, A_dB_last[0], comp_of_dB)
    L_ana = alg._semiblind_fisher_logL_dB_dB_svd(u_e_v_last[0], res.s, prior, A_dB_last[0], comp_of_dB)
    
    #return L_brute, L_param, L_ana
    return L_ana


#Added by Clement Leloup
def noise_real_max(components, instrument, templates, data, fsky, nside, invN, noise_seed, nblind, **minimize_kwargs):

    instrument = _force_keys_as_attributes(instrument)
    lmax = 3 * nside - 1
    n_comp = len(components)

    print('Computing prior')
    if isinstance(templates, np.ndarray):
        ell_in = np.arange(2, lmax+1)
        cl_in = np.concatenate((np.zeros((2, 2)), 2*np.pi*templates[1:3,:lmax-1]/(ell_in*(ell_in+1))), axis=1) #Take only polarization hard coded here
    else:
        cl_in = hp.read_cl(templates)[1:3,:lmax+1] #Take only polarization
        
    #Format the prior shape
    with np.errstate(divide='ignore'):
        EE_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[0,:]]) #Should modify here in case several non-blind components
        BB_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[1,:]])
        cl_in = np.stack((EE_in, BB_in), axis=1) #Probably a better way to do that
    cl_in[~np.isfinite(cl_in)] = 0.
    

    print('Computing mixing matrix')
    A_ev, A_dB_ev, comp_of_param, x0, params = init_semiblind_mixmat(components, instrument, nblind)
    if not len(x0):
        print("Nothing to maximize !")
        return x0

    print('x0 : ', x0)
        
    print('Computing alms')
    try:
        assert np.any(instrument.Beams)
    except (KeyError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams
    alms = _get_alms(data, beams, lmax)[:,1:,:]
    ell = hp.Alm.getlm(lmax, np.arange(alms.shape[-1]))[0]
    ell = np.stack((ell, ell), axis=-1).reshape(-1)
    lmin = 30
    mask_lmin = [l < lmin for l in ell]

    #Add noise to data alms
    np.random.seed(noise_seed)
    nlms_E = [hp.synalm(alg._inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms_B = [hp.synalm(alg._inv(invN)[:, f, f], lmax) for f in np.arange(invN.shape[1])]
    nlms = np.concatenate((np.asarray(nlms_E)[:,np.newaxis,:], np.asarray(nlms_B)[:,np.newaxis,:]), axis=1)
    alms = alms+nlms
    
    #Produce alms from maps
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    alms[..., np.arange(1, lmax+1, 2)] = hp.UNSEEN  # Mask imaginary m = 0
    mask_alms = _intersect_mask(alms)
    alms[..., mask_alms] = 0  # Thus no contribution to the spectral likelihood
    alms = np.swapaxes(alms, 0, 2)
    alms[mask_lmin, ...] = 0
    
    prior = np.array([cl_in[l,:,:] for l in ell])/fsky
    invNlm = np.array([invN[l,:,:] for l in ell])[:,np.newaxis,:,:]
    
    #Maximize likelihood
    A_dB_ev, comp_of_dB = alg._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)
    fun, jac, last_values = alg._semiblind_build_bound_inv_logL_and_logL_dB(
        A_ev, alms, invNlm, prior, A_dB_ev, comp_of_dB) #modification here
    minimize_kwargs['jac'] = jac
    res = sp.optimize.minimize(fun, x0, **minimize_kwargs)
    m = res.x
    print('xmax = ', res.x)

    return m


#Added by Clement Leloup
def init_semiblind_mixmat(components, instrument, nblind, prewhiten_factors=None):
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument.Frequencies)
    x0 = np.array([x for c in components for x in c.defaults])
    if not len(x0):
        A_ev = A_ev()
        notblind = A_ev.shape[1]-nblind
        A1 = A_ev[notblind:A_ev.shape[1],notblind:]
        B = np.concatenate((A_ev[:,0:notblind, np.newaxis], np.dot(A_ev[:,notblind:], np.linalg.inv(A1))), axis=1)
    else: #modification by Clement Leloup
        A_ev = A_ev(x0)
        notblind = A_ev.shape[1]-nblind
        A1 = A_ev[notblind:A_ev.shape[1],notblind:]
        B_mat = np.concatenate((A_ev[:,0:notblind], np.dot(A_ev[:,notblind:], np.linalg.inv(A1))), axis=1)
        #print('B_mat = ', B_mat)

    #comp = [CMB()], SemiBlind(B_mat[:,1], 1), SemiBlind(B_mat[:,2], 2)]
    #comp = np.append([CMB()], [SemiBlind(B_mat[:,i], i) for i in np.arange(1, len(components))])
    comp = np.append(components[:notblind], [SemiBlind(B_mat[:,i], i, np.arange(notblind, len(components))) for i in np.arange(notblind, len(components))])
    B = MixingMatrix(*comp)
    B_ev = B.evaluator(instrument.Frequencies)
    B_dB_ev = B.diff_evaluator(instrument.Frequencies)
    comp_of_dB = B.comp_of_dB
    xB0 = np.array([x for c in comp for x in c.defaults])
    params = B.params
    #print('B = ', B_ev(xB0))

    if prewhiten_factors is None:
        return B_ev, B_dB_ev, comp_of_dB, xB0, params

    if B.n_param:
        pw_B_ev = lambda x: prewhiten_factors[..., np.newaxis] * B_ev(x)
        pw_B_dB_ev = lambda x: [prewhiten_factors[..., np.newaxis] * B_dB_i
                                for B_dB_i in B_dB_ev(x)]
    else:
        pw_B_ev = lambda: prewhiten_factors[..., np.newaxis] * B_ev()
        pw_B_dB_ev = None

    return pw_B_ev, pw_B_dB_ev, comp_of_dB, xB0, params

def harmonic_ilc(components, instrument, data, lbins=None, weights=None):
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
    nside = hp.get_nside(data)
    lmax = 3 * nside - 1
    lmax = min(lmax, lbins.max())
    n_comp = len(components)
    if weights is not None:
        assert not np.any(_intersect_mask(data) * weights.astype(bool))
        fsky = np.mean(weights**2)**2 / np.mean(weights**4)
    else:
        mask = _intersect_mask(data)
        fsky = float(mask.sum()) / mask.size


    print('Computing alms')
    try:
        assert np.any(instrument.Beams)
    except (KeyError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams

    alms = _get_alms(data, beams, lmax, weights)

    print('Computing ILC')
    res = _harmonic_ilc_alm(components, instrument, alms, lbins, fsky)

    print('Back to real')
    res.s = np.empty((n_comp,) + data.shape[1:], dtype=data.dtype)
    for c in range(n_comp):
        res.s[c] = hp.alm2map(alms[c], nside)

    return res


def _get_alms(data, beams=None, lmax=None, weights=None):
    alms = []
    for f, fdata in enumerate(data):
        if weights is None:
            alms.append(hp.map2alm(fdata, lmax=lmax))
        else:
            alms.append(hp.map2alm(hp.ma(fdata)*weights, lmax=lmax))
        print('%i of %i complete' % (f+1, len(data)))
    alms = np.array(alms)

    if beams is not None:
        print('Correcting alms for the beams')
        # FIXME correct polarization with polarization beams
        for fwhm, alm in zip(beams, alms):
            bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax)
            hp.almxfl(alm, 1.0/bl, inplace=True)

    return alms


def _harmonic_ilc_alm(components, instrument, alms, lbins=None, fsky=None):
    cl_in = np.array([hp.alm2cl(alm) for alm in alms])

    # Multipoles for the ILC bins
    lmax = hp.Alm.getlmax(alms.shape[-1])
    ell = hp.Alm.getlm(lmax, np.arange(alms.shape[-1]))[0]
    if lbins is not None:
        ell = np.digitize(ell, lbins)

    # Make alms real
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    alms[..., np.arange(1, lmax+1, 2)] = hp.UNSEEN  # Mask imaginary m = 0
    ell = np.stack((ell, ell), axis=-1).reshape(-1)

    res = ilc(components, instrument, alms, ell)

    # Craft output
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
    assert len(instrument.Frequencies) == data.shape[0]
    n_freq = data.shape[0]
    n_comp = len(components)
    n_id = patch_ids.max() + 1

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
        data_patch = data[mask_pix].reshape(-1, n_freq)
        np.einsum('ij,ik->jk', data_patch, data_patch,
                  out=res.freq_cov[mask_id])
        inv_freq_cov = np.linalg.inv(res.freq_cov[mask_id])
        res.W[mask_id] = alg.W(A, inv_freq_cov)
        res.s[mask_pix] = alg._mv(res.W[mask_id], data_patch)

    if patch_ids is None:
        res.freq_cov = np.full((n_freq, n_freq), hp.UNSEEN)
        res.W = np.full((n_comp, n_freq), hp.UNSEEN)
        ilc_patch(mask, np.s_[:])
    else:
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
