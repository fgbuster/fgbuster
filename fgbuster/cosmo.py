"""Routines to estimate constraints on cosmological parameters

"""
import numpy as np
import healpy as hp
import os.path as op
from fgbuster.xForecast import _get_Cl_cmb
import sys
import scipy

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

