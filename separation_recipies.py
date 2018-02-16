""" Component separation with many different setups

"""

def basic_comp_sep(components, instrument, data, nside=0):
    """ Basic component separation

    Parameters
    ----------
    components: list
        List of the `Components` in the data model
    instrument: PySM.Instrument
        Instrument object used to define the mixing matrix and the
        frequency-dependent noise weight.
        It is required to have:
          - frequencies
          - noise
        however, also the following are taken into account, if provided
          - bandpass
    data: array
        Data vector to be separated
    nside:
        For each pixel of a HEALPix map with this nside, the non-linear
        parameters are estimated independently
    """
    # Build A_ev
    # Build x0
    # Preapre the map 
    # Launch component separation
    # Reorganize the results
