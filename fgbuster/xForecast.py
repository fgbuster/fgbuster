""" Forecasting toolbox for DustBust
"""

#import cosmological_analysis as ca
import angular_spectrum_estimation as ase
from algebra import multi_comp_sep, comp_sep, W_dBdB, W_dB, _mm, _mtmm, W, _mmm, _utmv, _mmv
from .mixingmatrix import MixingMatrix
import numpy as np
import pylab as pl
import healpy as hp
import os.path as op
import scipy as sp
from scipy import optimize

CMB_CL_FILE = op.join(
    op.dirname(__file__), 'templates/ClCAMB_Planck15_lmax4200_%s.fits')

def xForecast(components, instrument, invN, d_fgs, lmin, lmax, fsky, Alens=1.0, r=0.001, estimator='', *minimize_args, **minimize_kwargs):
    """ Run xForecast or CMB4cast using the provided
       instrumental specifications and input foregrounds 
       maps 

    Parameters
    ----------
    ### TODO
    ### CLEAN THE FOLLOWING TEXT
    A_ev: function or list
        The evaluator of the mixing matrix. It takes a float or an array as
        argument and returns the mixing matrix, a ndarray with shape
        `(..., n_freq, n_comp)`
        If list, the i-th entry is the evaluator of the i-th patch.
    d_fgs: ndarray
        The data vector. Shape `(n_freq, n_stokes, n_pix)`.
    invN: ndarray or None
        The inverse noise matrix. Shape `(..., n_freq, n_freq)`.
    lmin: minimum multipole entering the likelihood computation
    lmax: maximum multipole entering the likelihood computation
    fsky: fraction of sky entering the likelihood computation
    Alens: A_lens, amplitude of the lensing B-modes entering the 
        likelihood on r. 
    r: tensor-to-scalar ratio, amplitude of the primordial B-modes entering the 
        simulations and likelihood on r. 
    estimator: power spectrum estimator to be chosen among ...... 
    minimize_args: list
        Positional arguments to be passed to `scipy.optimize.minimize`.
        At this moment it just contains `x0`, the initial guess for the spectral
        parameters
    minimize_kwargs: dict
        Keyword arguments to be passed to `scipy.optimize.minimize`.
        A good choice for most cases is
        `minimize_kwargs = {'tol': 1, options: {'disp': True}}`. `tol` depends
        on both the solver and your signal to noise: it should ensure that the
        difference between the best fit -logL and and the minimum is well less
        then 1, without exagereting (a difference of 1e-4 is useless).
        `disp` also triggers a verbose callback that monitors the convergence.

    Returns
    -------
    xFres:  
    """

    ### TODO [DAVIDE] DONE
    nside = hp.npix2nside(d_fgs.shape[-1])
    nside_patch = 0
    cl_cmb = _get_Cl_cmb( Alens=Alens, r=r)
    s_cmb = hp.synfast(cl_cmb, nside=nside)
    print s_cmb.shape
    if d_fgs.shape[1] == 2:
        s_cmb = s_cmb[1:,:]
    elif d_fgs.shape[1] == 1:
        s_cmb = s_cmb[0,:]
    ###############################################################################
    # 0. Prepare noise-free "data sets"
    d_obs = d_fgs.T + (components[0].eval(instrument.Frequencies)*s_cmb[...,np.newaxis]).swapaxes(-3,-2)

    ###############################################################################
    # 1. Component separation using the noise-free data sets
    # grab the max-L spectra parameters with the associated error bars
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument.Frequencies)
    A_dB_ev = A.diff_evaluator(instrument.Frequencies)
    A_dBdB_ev = A.diff_diff_evaluator(instrument.Frequencies)

    x0 = np.array([x for c in components for x in c.defaults])
    params = A.params
    if nside_patch == 0:
        res = comp_sep(A_ev, d_obs, invN, A_dB_ev, A.comp_of_dB, x0,
                       options=dict(disp=True))
    else:
        patch_ids = hp.ud_grade(np.arange(hp.nside2npix(nside)),
                                hp.npix2nside(d_obs.shape[-1]))
        res = multi_comp_sep(A_ev, d_obs, invN, patch_ids, x0)

    res.params =  params
    res.s = res.s.T
    A_maxL = A_ev(res.x)
    A_dB_maxL = A_dB_ev(res.x)
    A_dBdB_maxL = A_dBdB_ev(res.x)

    ###############################################################################
    # 2. Estimate noise after component separation
    ### TO DO [DAVIDE] DONE
    ### A^T N_ell^-1 A
    ind_cmb = A.components.index('CMB')
    Cl_noise = _get_Cl_noise(instrument, A_maxL, lmax)[ind_cmb,ind_cmb,lmin-2:lmax-2]

    ###############################################################################
    # 3. Compute spectra of the input foregrounds maps
    ### TO DO: which size for Cl_fgs??? N_spec != 1 ? 
    N_freqs = d_fgs.shape[0]
    for f_1 in range(N_freqs):
        for f_2 in range(N_freqs):
            if f_2 >= f_1:
                # we only take the BB spectra, for ell >= 2
                # this form should be able to handle binned spectra as well
                Cl_loc, ell = ase.TEB_spectra( d_fgs[f_1,:,:], IQU_map_2=d_fgs[f_2,:,:], lmax=lmax, estimator=estimator)
                if f_1 == 0 and f_2 == 0:
                    Cl_fgs = np.zeros((N_freqs, N_freqs, len(Cl_loc[2])))
                Cl_fgs[f_1,f_2,:] = Cl_loc[2]*1.0
            else:
                # symmetrization of the Cl_fgs matrix
                Cl_fgs[f_1,f_2,:] = Cl_fgs[f_2,f_1,:]
    Cl_fgs = Cl_fgs.swapaxes(-1,0)
    ell = ell[lmin-2:lmax-2]
    ###############################################################################
    # 4. Estimate the statistical and systematic foregrounds residuals 

    ### find ind_cmb, the dimension of the CMB component
    ### TO DO [DAVIDE] DONE
    ### add this list to the MixingMatrix class 
    ind_cmb = A.components.index('CMB')
    W_maxL = W(A_maxL, invN=invN)[...,ind_cmb,:]
    W_dB_maxL = W_dB(A_maxL, A_dB_maxL, A.comp_of_dB, invN=invN)[...,ind_cmb,:]
    W_dBdB_maxL = W_dBdB(A_maxL, A_dB_maxL, A_dBdB_maxL, A.comp_of_dB, invN=invN)[...,ind_cmb,:]
    V_maxL = np.einsum('ij,ij...->...', res.Sigma, W_dBdB_maxL )

    # elementary quantities defined in Stompor, Errard, Poletti (2016)
    Cl_xF = {}
    Cl_xF['yy'] = _utmv(W_maxL, Cl_fgs, W_maxL)
    Cl_xF['YY'] = _mmm(W_dB_maxL, Cl_fgs, W_dB_maxL.T)
    Cl_xF['yz'] = _utmv(W_maxL, Cl_fgs, V_maxL )
    Cl_xF['Yy'] = _mmv(W_dB_maxL, Cl_fgs, W_maxL)
    Cl_xF['Yz'] = _mmv(W_dB_maxL, Cl_fgs, V_maxL)
    Cl_xF['zY'] = Cl_xF['Yz'].T
    Cl_xF['zy'] = Cl_xF['yz']
    Cl_xF['yY'] = Cl_xF['Yy'].T
    for key in Cl_xF.keys():
        Cl_xF[key] = Cl_xF[key][lmin-2:lmax-2]
    # bias and statistical foregrounds residuals
    res.bias = Cl_xF['yy'] + Cl_xF['yz'] + Cl_xF['zy']
    YSY =  np.sum(np.sum(res.Sigma*Cl_xF['YY'], axis=-1), axis=-1)
    res.stat = np.trace( _mm(res.Sigma, Cl_xF['YY']), axis1=-2, axis2=-1 )
    res.var = 2*(_mmm(Cl_xF['yY'].T, res.Sigma, Cl_xF['Yy'].T ) + res.stat** 2)
    res.noise = Cl_noise*1.0

    ###############################################################################
    # 5. Plug into the cosmological likelihood
    Cl_fid = {}
    Cl_fid['BB'] = cl_cmb[2][lmin-2:lmax-2]
    Cl_fid['BuBu'] = _get_Cl_cmb(Alens=0.0, r=1.0)[2][lmin-2:lmax-2]
    Cl_fid['BlBl'] = _get_Cl_cmb(Alens=1.0, r=0.0)[2][lmin-2:lmax-2]

    ## 5.1. data 
    E = np.diag(Cl_fid['BB'] + YSY + Cl_xF['yy'] + Cl_xF['zy'] + Cl_xF['yz'])
    ## 5.2. modeling
    def cosmo_likelihood(r_):
        Cl_BB_model = Cl_fid['BlBl']*Alens + Cl_fid['BuBu']*r_+ Cl_noise
        U_inv = _mm(res.Sigma_inv, np.sum( (2*ell+1)*Cl_xF['YY'].swapaxes(0,-1)/Cl_BB_model, axis=-1) )
        U = np.linalg.inv( U_inv ) 
        
        term_0 = (2*ell+1)*(1.0 - (1.0/Cl_BB_model)*np.trace(_mm(U, Cl_xF['YY']), axis1=-2, axis2=-1))
        term_1 = ((2*ell+1)/Cl_BB_model)*np.trace(_mm(res.Sigma,Cl_xF['YY']), axis1=-2, axis2=-1)
        trCinvC_1 = np.sum( Cl_fid['BB']/Cl_BB_model*term_0 + term_1 )
        
        trCinvC_2 = 0.0
        for i in range(len(ell)):
            for j in range(len(ell)):
                trCinvC_2 += ((2*ell[i]+1)/Cl_BB_model[i])*((2*ell[j]+1)/Cl_BB_model[j])*\
                       np.trace(_mm(_mm(U, Cl_xF['YY'][j,:,:]), _mm(res.Sigma, Cl_xF['YY'][i,:,:].T)), axis1=-2, axis2=-1)
       
        trCinvEC_1 = np.sum( ((2*ell+1)/Cl_BB_model)*(Cl_xF['yy'] + Cl_xF['zy'] + Cl_xF['yz']) )
       
        trCinvEC_2 = 0.0
        for i in range(len(ell)):
            for j in range(len(ell)):
                trCinvEC_2 += ((2*ell[i]+1)/Cl_BB_model[i])*((2*ell[j]+1)/Cl_BB_model[j])*\
                       np.trace( U.dot(Cl_xF['YY'][j,:,:].dot(res.Sigma.dot(Cl_xF['YY'][i,:,:]))), axis1=-2, axis2=-1)

        trCE = trCinvC_1 - trCinvC_2 + trCinvEC_1 - trCinvEC_2

        D = np.diag( Cl_BB_model )
        logdetC = np.real(np.trace( (2*ell+1)*sp.linalg.logm( D )) + \
                    np.trace( sp.linalg.logm(res.Sigma) ) -\
                    np.trace( sp.linalg.logm(U) ))

        logL = fsky*( trCE + logdetC ) 
        return logL

    ### TODO [JOSQUIN]
    ###  minimization, gridding, sigma(r)
    # Likelihood maximization
    r0 = 1e-3
    # r_v = np.logspace(-4,2, num=50)
    # logL = []
    # for r_loc in r_v:
    #     logL.append(cosmo_likelihood(r_loc))
    # pl.figure()
    # pl.semilogx(r_v, logL, 'k-')
    # pl.show()
    # exit()

    res_Lr = sp.optimize.minimize(cosmo_likelihood, [r0], bounds=[(0,1e5)], *minimize_args, **minimize_kwargs)
    print ' res_Lr = ', res_Lr

    def sigma_r_computation_from_logL(r_loc):
        THRESHOLD = 1.00
        # THRESHOLD = 2.30 when two fitted parameters
        delta = np.abs( cosmo_likelihood(r_loc) - res_Lr['fun'] - THRESHOLD )
        return delta

    sr0 = 1e-3
    res_sr = sp.optimize.minimize(sigma_r_computation_from_logL, [sr0], bounds=[(0,1e5)], *minimize_args, **minimize_kwargs)
    print ' res_sr = ', res_sr
    res.cosmo_params = {}
    res.cosmo_params['r'] = (res_Lr['x'],res_sr['x'])

    ### TODO [DAVIDE]        
    ### outputs
    # possibility of returning a gridded likelihood?

    ###############################################################################
    # 6. Produce figures
    '''
        # angular power spectrum showing theoretical Cl / noise per freq band / noise after comp sep / stat and sys residuals
        # the emcee panels for the spectral parameters fit
        # the profile cosmological likelihood (if it has been gridded)
    '''

def _get_Cl_cmb(Alens=1., r=0.):
    power_spectrum = hp.read_cl(CMB_CL_FILE%'scalar')
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r != 0.:
        power_spectrum += r * hp.read_cl(CMB_CL_FILE%'tensor')
    return power_spectrum


def _get_Cl_noise(instrument, A, lmax):
    bl = [hp.gauss_beam(np.radians(b/60.), lmax=lmax) for b in instrument.Beams]
    nl = (np.array(bl) / np.radians(instrument.Sens_P/60.)[:, np.newaxis])**2
    AtNA = np.einsum('...fi,...fl,...fj->...lij', A, nl, A)
    inv_AtNA = np.linalg.inv(AtNA)
    return inv_AtNA.swapaxes(-3, -1)
