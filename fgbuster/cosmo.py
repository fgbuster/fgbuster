"""Routines to estimate constraints on cosmological parameters

"""
import numpy as np
import healpy as hp
import os.path as op
from fgbuster.xForecast import _get_Cl_cmb
import sys
import scipy
from fgbuster.algebra import W_dB, _mmm

CMB_CL_FILE = op.join(
     op.dirname(__file__), 'templates/Cls_Planck2018_%s.fits')


def from_Cl_to_r_estimate(ClBB_tot, ell_v, fsky, ClBB_model_other_than_prim, r_v, **minimize_kwargs):
    """Simple cosmological likelihood

    Parameters
    ----------
    ClBB_tot: array
         1-d vector of `C_\ell^BB` that corresponds to the total power in the analyzed map 
         It could correspond to a binned spectrum, as long as the matching multipoles 
         are described in the next input, `ell_v`
    ell_v: array
         1-d vector of multipoles `\ell`, of size matching `ClBB_tot`, giving the multipoles
         where ClBB_tot is defined
    fsky: float
         fraction of the sky to be considered in the cosmological likelihood, used
         to compute the number of degrees of freedom
    ClBB_model_other_than_prim: ndarray
         Vector of angular powers, of the same size as `ClBB_tot`.
         The modeled covariance being written as `ClBB_tot_modeled = ClBB_prim(r) + ClBB_model_other_than_prim`
         Typically, `ClBB_model_other_than_prim` contains noise, lensing and possibly an estimate
         of foregrounds residuals.
    r_v: array
         1-d vector of values of tensor-to-scalar ratio, over which
         the likelihood will be estimated. Use a fine enough grid as the recovered
         r value, as well as the sigma(r), are estimated using this grid

    Returns
    -------
    r_fit : float
         This is the tensor-to-scalar ratio from `r_v` at which the likelihood peaks.
    sigma_r_fit: float
         This is an estimate of the 1-sigma error bar on `r_fit`, including cosmic variance
         from primordial and lensing B-modes, but also noise and foregrounds residuals -- ie
         anything which is in `ClBB_tot`
    likelihood_on_r: array
         This 1-d vector contains the estimate of the likelihood at each element 
         of the `r_v` input vector.

    Note
    ----
      * Covariances are assumed diagonal in {ell,ell'}. This is OK for the average cases,
        and for some level of foregrounds residuals --- see e.g. Errard and Stompor 2018 
        for some discussions about it.
    """

    def likelihood_on_r_computation( r_loc, make_figure=False ):
        '''
        -2logL = sum_ell [ (2l+1)fsky * ( log(C) + C^-1.D  ) ]
            cf. eg. Tegmark 1998
        '''    
        Cov_model = np.diag( Cl_BB_prim*r_loc + ClBB_model_other_than_prim)
        logL = np.sum( (2*ell_v+1)*fsky*( np.log( Cov_model ) + ClBB_tot/Cov_model ))

        return logL
    
    # gridding -2log(L)
    logL = r_v*0.0
    for ir in range(len(r_v)):
        logL[ir] = likelihood_on_r_computation( r_v[ir] )
        ind = ir*100.0/len(r_v)
        sys.stdout.write("\r  .......... gridding the likelihood on tensor-to-scalar ratio >>>  %d %% " % ind )
        sys.stdout.flush()
    sys.stdout.write("\n")

    # renormalizing logL 
    chi2 = (logL - np.min(logL))/2.0
    # computing the likelihood itself, for plotting purposes
    likelihood_on_r = np.exp( - chi2 )/np.max(np.exp( - chi2 ))
    # estimated r is given by:
    r_fit = r_v[np.argmin(logL)]
    # and the 1-sigma error bar by (numerical recipies)
    ind_sigma = np.argmin(np.abs( (logL[np.argmin(logL):] - logL[np.argmin(logL)])/2.0 - 1.00/2.0 ))    
    sigma_r_fit =  r_v[ind_sigma+np.argmin(logL)] - r_fit

    return r_fit, sigma_r_fit, likelihood_on_r


def estimation_of_Cl_stat_res(Sigma, d_fgs, A_ev, A_dB_ev, comp_of_dB, beta_maxL, invN, lmin, lmax, i_cmb=0):

    """Estimation of the statistical residuals
    following Errard and Stompor 2018.

    Parameters
    ----------
    Sigma: 2d-array or list of 2d-arrays
         Covariance of error bars on spectral indices.
    d_fgs: ndarray
         The data vector. Shape `(..., n_freq)`.
    A_ev : function
        The evaluator of the mixing matrix. It takes a float or an array as
        argument and returns the mixing matrix, a ndarray with shape
        `(..., n_freq, n_comp)`
    invN: ndarray or None
        The inverse noise matrix. Shape `(..., n_freq, n_freq)`.
    A_dB_ev : function
        The evaluator of the derivative of the mixing matrix.
        It returns a list, each entry is the derivative with respect to a
        different parameter.
    comp_of_dB: IndexExpression or list of IndexExpression
        It allows to provide in `A_dB` only the non-zero columns `A`.
        `A_dB` is assumed to be the derivative of `A[comp_of_dB]`.
        If a list is provided, also `A_dB` has to be a list and
        `A_dB[i]` is assumed to be the derivative of `A[comp_of_dB[i]]`.
    beta_maxL: 1-d array or list of 1-d arrays
         Spectral indices as found by the maximization of the
         spectral likelihood     
    lmin, lmax: int
         minimum and maximum multipoles to be considered 
         in the analysis
    i_cmb: int
         Index for the CMB dimensions, typically given by
         A.components.index('CMB'), with A = MixingMatrix(*components)
    Returns
    -------
    ClBB_stat_model: ndarray
         Vector containing the amplitudes of the angular power spectrum

    Note
    ----
         * The formalism is following Errard and Stompor 2018
         * In the case of multipatch, this assumes uncorrelated foregrounds residuals between sky patches

    """

    # first, we preprocess the frequency sky maps
    n_stokes = d_fgs.shape[1]
    n_freqs = d_fgs.shape[0]
    if isinstance(Sigma, (list,)):
        n_patches = len(Sigma)
    else: n_patches = 1
    
    if n_stokes == 3:  
        d_spectra = d_fgs
    else:  # Only P is provided, add T for map2alm
        d_spectra = np.zeros((n_freqs, 3, d_fgs.shape[2]), dtype=d_fgs.dtype)
        d_spectra[:, 1:] = d_fgs
    # masked data and fsky
    mask = d_fgs[0, 0, :] != 0.
    fsky = mask.astype(float).sum() / mask.size
    # go to Fourier space and build matrix with
    # all the observed auto- and cross-spectra 
    almBs = [hp.map2alm(freq_map, lmax=lmax, iter=10)[2] for freq_map in d_spectra]
    Cl_fgs = np.zeros((n_freqs, n_freqs, lmax+1), dtype=d_fgs.dtype)
    for f1 in range(n_freqs):
        for f2 in range(n_freqs):
            if f1 > f2:
                Cl_fgs[f1, f2] = Cl_fgs[f2, f1]
            else:
                Cl_fgs[f1, f2] = hp.alm2cl(almBs[f1], almBs[f2], lmax=lmax)
    # concatanating everything, and correcting for fsky
    Cl_fgs = Cl_fgs[..., lmin:] / fsky
    # from the mixing matrix and its derivative at the peak
    # of the likelihood, we build dW/dB
    A_maxL = A_ev(beta_maxL)
    A_dB_maxL = A_dB_ev(beta_maxL)
    W_dB_maxL = W_dB(A_maxL, A_dB_maxL, comp_of_dB, invN=invN)[:, i_cmb]
    # and then Cl_YY, cf Stompor et al 2016
    Cl_YY = _mmm(W_dB_maxL, Cl_fgs.T, W_dB_maxL.T)  
    # and finally, using the covariance of error bars on spectral indices
    # we compute the model for the statistical foregrounds residuals, 
    # cf. Errard et al 2018
    tr_SigmaYY = np.einsum('ij, lji -> l', Sigma, Cl_YY)

    return tr_SigmaYY
