""" Forecasting toolbox for DustBust
"""

import cosmological_analysis as ca
import angular_spectrum_estimation as ase
from algebra import multi_comp_sep, comp_sep
import separation_recipies as sr
import numpy as np
import pylab as pl
import healpy as hp

def xForecast(A_ev, invN, data, s_cmb_true, estimator=''):
   """ Run xForecast or CMB4cast using the provided
       instrumental specifications and input foregrounds 
       maps 

    Parameters
    ----------
    A_ev: function or list
        The evaluator of the mixing matrix. It takes a float or an array as
        argument and returns the mixing matrix, a ndarray with shape
        `(..., n_freq, n_comp)`
        If list, the i-th entry is the evaluator of the i-th patch.
    data: ndarray
        The data vector. Shape `(..., n_freq)`.
    invN: ndarray or None
        The inverse noise matrix. Shape `(..., n_freq, n_freq)`.
    s_cmb_true: input pure CMB map to estimate foregrounds residuals
        in the recovered CMB map
    estimator: power spectrum estimator to be chosen among ...... 
	Cl_theory: theoretical angular power spectrum for likelihood

    Returns
    -------
    xFres:  
    """

    # 0. Prepare noise-free "data sets"
    prewhiten_factors = sr._get_prewhiten_factors(instrument, data.shape)
    A_ev, A_dB_ev, comp_of_param, params = sr._build_A_evaluators(
        components, instrument, prewhiten_factors=prewhiten_factors)
    x0 = np.array([x for c in components for x in c.defaults])
    prewhitened_data = prewhiten_factors * data.T
    
    ###############################################################################
    # 1. Component separation using the noise-free data sets
    # grab the max-L spectra parameters with the associated error bars
    if nside == 0:
        res = comp_sep(A_ev, prewhitened_data, None, A_dB_ev, comp_of_param, x0,
                       options=dict(disp=True))
    else:
        patch_ids = hp.ud_grade(np.arange(hp.nside2npix(nside)),
                                hp.npix2nside(data.shape[-1]))
        res = multi_comp_sep(A_ev, prewhitened_data, None, patch_ids, x0)
    res.params = params
    res.s = res.s.T

    ###############################################################################
    # 2. Estimate noise after component separation
    Cl_noise = sr.Cl_noise_builder( instrument, res )

    ###############################################################################
    # 3. Compute spectra of the input foregrounds maps
    Cl_fgs = ase.TEB_spectra( data, estimator=estimator )

    ###############################################################################
    # 4. Estimate the statistical and systematic foregrounds residuals 
    '''
    w_loc = AtNAinv_fit['matrix'].dot( A_fit['matrix'].T ).dot( Ninv_p['matrix'] )
    -----------
    AtNAinv_fit = AtNAinv_builder( A_fit, Ninv_p )
    dAtNAinv_fit = - AtNAinv_fit['matrix'].dot( dAdB_fit[drv_loc]['matrix'].T.dot( Ninv_p['matrix'] ).dot( A_fit['matrix'] ) +\
                  A_fit['matrix'].T.dot( Ninv_p['matrix'] ).dot( dAdB_fit[drv_loc]['matrix'] )).dot( AtNAinv_fit['matrix'] )
    dW_loc = dAtNAinv_fit.dot( A_fit['matrix'].T ).dot( Ninv_p['matrix'] ) +\
              AtNAinv_fit['matrix'].dot( dAdB_fit[drv_loc]['matrix'].T ).dot( Ninv_p['matrix'] )
    -----------
    dAtNAinv_fit_sq_i = - AtNAinv_fit['matrix'].dot( dAdB_fit[drv[i]]['matrix'].T.dot(Ninv_p['matrix']).dot(A_fit['matrix']) +\
                        A_fit['matrix'].T.dot(Ninv_p['matrix']).dot( dAdB_fit[drv[i]]['matrix']) ).dot( AtNAinv_fit['matrix'] )		
    # derivative of (ATNA)^-1 wrt. beta_j
    dAtNAinv_fit_sq_j = - AtNAinv_fit['matrix'].dot( dAdB_fit[drv[j]]['matrix'].T.dot(Ninv_p['matrix']).dot(A_fit['matrix']) +\
                         A_fit['matrix'].T.dot(Ninv_p['matrix']).dot( dAdB_fit[drv[j]]['matrix']) ).dot( AtNAinv_fit['matrix'] )
    # derivative of (ATNA)^-1 wrt. beta_i, beta_j
    ddAtNAinv_fit_sq = - dAtNAinv_fit_sq_j.dot( dAdB_fit[drv[i]]['matrix'].T.dot( Ninv_p['matrix'] ).dot( A_fit['matrix'] ) +\
                     A_fit['matrix'].T.dot(Ninv_p['matrix']).dot(dAdB_fit[drv[i]]['matrix'])  ).dot( AtNAinv_fit['matrix'] ) -\
                     AtNAinv_fit['matrix'].dot( dAdBdB_fit[drv[j]][drv[i]]['matrix'].T.dot(Ninv_p['matrix']).dot(A_fit['matrix']) + \
                     dAdB_fit[drv[i]]['matrix'].T.dot(Ninv_p['matrix']).dot(dAdB_fit[drv[j]]['matrix'])+\
                     dAdB_fit[drv[j]]['matrix'].T.dot(Ninv_p['matrix']).dot(dAdB_fit[drv[i]]['matrix'])+\
                     A_fit['matrix'].T.dot(Ninv_p['matrix']).dot(dAdBdB_fit[drv[j]][drv[i]]['matrix'])).dot( AtNAinv_fit['matrix'] ) -\
                     AtNAinv_fit['matrix'].dot( dAdB_fit[drv[i]]['matrix'].T.dot( Ninv_p['matrix'] ).dot( A_fit['matrix'] ) +\
                     A_fit['matrix'].T.dot(Ninv_p['matrix']).dot(dAdB_fit[drv[i]]['matrix'])).dot(dAtNAinv_fit_sq_j)
    # entire derivative
    d2Wdbdb_loc =  ddAtNAinv_fit_sq.dot( A_fit['matrix'].T ).dot(Ninv_p['matrix']) + \
                 2*dAtNAinv_fit_sq_i.dot( dAdB_fit[drv[j]]['matrix'].T ).dot( Ninv_p['matrix'] ) +\
                 AtNAinv_fit['matrix'].dot( dAdBdB_fit[drv[j]][drv[i]]['matrix'].T ).dot( Ninv_p['matrix'] )
    -----------
    vk['matrix'][:,p] += d2LdBdBinv_loc['matrix'][k1,k2]*d2Wdbdb['matrix'][k1,k2,:,p]
    -----------
    Cl_yy[ell_ind] = wk['matrix'].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot(wk['matrix']) 
    Cl_YY[ch1, ch2, ell_ind] = dW_kb['matrix'][ch1,:].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot(dW_kb['matrix'][ch2,:]) 
    Cl_yz[ell_ind] = wk['matrix'].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot(vk['matrix']) 
    Cl_Yy[ch,ell_ind] = dW_kb['matrix'][ch,:].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot( wk['matrix'] ) 
    Cl_Yz[ch,ell_ind] = dW_kb['matrix'][ch,:].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot( vk['matrix'] ) 
    Cl_zY[ch,ell_ind] = vk['matrix'].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot(  dW_kb['matrix'][ch,:] ) 
    Cl_zy[ell_ind] = vk['matrix'].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot(wk['matrix']) 
    Cl_yY[ch,ell_ind] =  wk['matrix'].T.dot(Fl_fgs['matrix'][:,:,ell_ind]).dot( dW_kb['matrix'][ch,:] )
    -----------
    first_terms = Cls['yy'][ell_ind] + Cls['yz'][ell_ind] + Cls['zy'][ell_ind]
    Cl_res_bias[ell_ind] = np.abs(first_terms)
    trace_term = np.trace( d2LdBdBinv_loc['matrix'].dot(Cls['YY'][:,:, ell_ind]) )
    Cl_res_stat[ell_ind] = np.abs(trace_term)
    Cl_res_var[ell_ind]  = np.abs(2*Cls['yY'][:,ell_ind].T.dot( d2LdBdBinv_loc['matrix'][:,:].dot(Cls['Yy'][:,ell_ind]) ) \
                          + 2*Cl_res_stat[ell_ind] ** 2)
    '''
    

    ###############################################################################
    # 5. Plug into the cosmological likelihood
    '''
    E = np.diag( (C_ell_fid[ell_min-2:ell_max-1] + YSY \
       + Cls['yy'][ell_min-2:ell_max-1] \
       + Cls['zy'][ell_min-2:ell_max-1] \
       + Cls['yz'][ell_min-2:ell_max-1] ))
    -----------
    U_inv[k1,k2] = Sigma_inv[k1,k2] + \
                   np.sum( (2*ell_v[:] + 1)*Cls['YY'][k1,k2,ell_min-2:ell_max-1]/C_ell[:] )
    -----------
    first_term = (2*ell_v[:]+1)*( 1 - ( 1.0/Cl )*np.trace( U.dot(Cls['YY'][:,:,ell_min-2:ell_max-1]) ) )
    second_term = ((2*ell_v[:]+1)/Cl)*np.trace(d2LdBdB_loc_matrix.dot(Cls['YY'][:,:,ell_min-2:ell_max-1]))
    for ell_ind1 in range(max_ell):
        for ell_ind2 in range(max_ell):
            trCinvC_2 += ((2*ell_v[ell_ind1]+1)/Cl[ell_ind1])*((2*ell_v[ell_ind2]+1)/Cl[ell_ind2])*\
                       np.trace( U.dot(Cls['YY'][:,:,ell_min-2+ell_ind2].dot(d2LdBdB_loc_matrix.dot(Cls['YY'][:,:,ell_min-2+ell_ind1]))))
    -----------
    trCinvEC_1 = np.sum( ((2*ell_v[:]+1)/Cl)*( Cls['yy'][ell_min-2:ell_max-1] + Cls['zy'][ell_min-2:ell_max-1] + Cls['yz'][ell_min-2:ell_max-1] ) )
    -----------
    for ell_ind1 in range(max_ell):
        for ell_ind2 in range(max_ell):
            trCinvEC_2 += ((2*ell_v[ell_ind1]+1)/Cl[ell_ind1])*((2*ell_v[ell_ind2]+1)/Cl[ell_ind2])*\
                          np.trace( U.dot( Cls['Yy'][np.newaxis,:,ell_min-2+ell_ind2].T.dot(Cls['yY'][np.newaxis,:,ell_min-2+ell_ind1]) \
                          + Cls['Yy'][np.newaxis,:,ell_min-2+ell_ind2].T.dot(Cls['zY'][np.newaxis,:,ell_min-2+ell_ind1])\
                          + Cls['Yz'][np.newaxis,:,ell_min-2+ell_ind2].T.dot(Cls['yY'][np.newaxis,:,ell_min-2+ell_ind1])  ) )
    -----------
    trCE = trCinvC_1 - trCinvC_2 + trCinvEC_1 - trCinvEC_2
    D = np.diag( Cl )
    logdetC = np.trace( (2*ell_v+1)*scipy.linalg.logm( D )) + np.trace( scipy.linalg.logm(d2LdBdB_loc_matrix) ) -\
               np.trace( scipy.linalg.logm(U) )
    logL = fsky*( trCE + logdetC ) 
    -----------
    # analytical derivative of logL also available in xForecast .... 

    # optimization of logL wrt. r
    # estimation of the Hessian (analytical Fisher?) at the peak 
    # possibility of returning a gridded likelihood?
    '''

    ###############################################################################
    # 6. Produce figures
    '''
	# angular power spectrum showing theoretical Cl / noise per freq band / noise after comp sep / stat and sys residuals
	# the emcee panels for the spectral parameters fit
	# the profile cosmological likelihood (if it has been gridded)
    '''
