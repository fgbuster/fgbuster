""" Toolbox in order to transform maps to harmonic domain
The toolbox should eventually include the possibility to estimate power spectra with
    - Anafast
    - Xpol
    - Xpure
    - NaMaster
    - PolSpice?
"""

import numpy as np
import healpy as hp
import sys

def TEB_spectra( IQU_map, IQU_map_2=None, ell_max=0.0, estimator=None, *args, **kwargs ):
    """ Get a pre-defined PySM sky

    Parameters
    ----------
    IQU_map: tuple
             contains the input I, Q and U maps to be Fourier transformed
    IQU_map_2: tuple
               contains a possible second set of input I, Q and U maps to consider
               in the evaluation of cross-spectra
    ell_max: int
             maximum multipole to consider. By default, we consider the Nyquist limit
             i.e. ell_max = 2*nside of the input maps
    estimator: string
               choice of the power spectrum estimator, among 'anafast' (default) and NaMaster 
               the user can provide the necessary *args and **kwargs for each method 
               - nothing required for Anafast
               - w (NaMaster work space), mask (the binary mask) and mask_apo (the apodized mask)
                 apotype ('C1' by default), apodization_size (30 arcmin by default), nlb (number of ell bins)
                 if not provided, we build the necessary quantities in the code

    Returns
    -------
    sky: tuple containing the 6 angular spectra (ClTT, ClEE, ClBB, ClTE, ClTB, ClEB )
    """

    if estimator=='NaMaster':

        # XXX: to be debugged and validated
        # print("power spectrum estimation with NaMaster not ready yet", file=sys.stderr)
        print >> sys.stderr, "power spectrum estimation with NaMaster not ready yet"
        #--------------- 
        # import pymaster as nmt
        # if not w: 
        #     print('building mode coupling matrix as it is not provided')
        #     b=nmt.NmtBin( hp.npix2nside(IQU_map.shape[1]), nlb=nlb )
        #     w=nmt.NmtWorkspace()
        #     w.compute_coupling_matrix( Q, U, b )
        # if not mask_apo:
        #     if not mask:
        #         mask = np.ones(IQU_map[0].shape[0])
        #         mask[np.where(IQU_map[0] == 0)[0]] = 0.
        #     mask_apo = nmt.mask_apodization(mask, apodization_size, apotype=apotype)
        # f2 = nmt.NmtField(mask_apo, [Q,U], purify_e=True, purify_b=True)
        # field_0 = f2
        # if ((len(Q2)>1) and (len(U2)>1)):
        #     f2_2 = nmt.NmtField(mask_apo,[Q2,U2], purify_e=True, purify_b=True)
        #     field_1 = f2_2
        # else: field_1 = f2
        # cl_coupled=nmt.compute_coupled_cell( field_0, field_1 )
        # cl_decoupled=w.decouple_cell( cl_coupled )
        # return cl_decoupled

    elif (estimator=='Xpol' or estimator=='Xpure'):

        # XXX: to be written, using something close to what Julien Peloton
        #      did for S4CMB
        # print("power spectrum estimation with Xpol/Xpure not ready yet", file=sys.stderr)
        print >> sys.stderr, "power spectrum estimation with Xpol/Xpure not ready yet"
        #--------------- 
        # write_maps_a_la_xpure(sky_out_tot, name_out=name_out,
        #                           output_path='xpure/maps')
        # write_weights_a_la_xpure(sky_out_tot, name_out=name_out,
        #                              output_path='xpure/masks')
        
        # params_xpure = import_string_as_module(args.inifile_xpure)

        # batch_file = 'sim{:03d}_{}_{}_{}.batch'.format(
        #             args.sim_number,
        #             params.tag,
        #             params.name_instrument,
        #             params.name_strategy)
        
        # create_batch(batch_file, name_out, params, params_xpure)

        # qsub = commands.getoutput('sbatch ' + batch_file)
        # print(qsub)

    else:
        if ell_max <= 2*hp.npix2nside(IQU_map[0].shape[0]):
        	ell_max = 2*hp.npix2nside(IQU_map[0].shape[0])

        ClTT, ClEE, ClBB, ClTE, ClTB, ClEB = hp.sphtfunc.anafast( map1=IQU_map,map2=IQU_map_2,\
                                                                      iter=10, lmax=ell_max ) 

        return ClTT[2:], ClEE[2:], ClBB[2:], ClTE[2:], ClTB[2:], ClEB[2:]
