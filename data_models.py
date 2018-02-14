""" Tools for a handy building of the comp sep inputs (i.e. A_ev & invN)
"""

class AnalyticMixingMatrix(*analytic_components):
    """ As the analytic components, provide an evaluator of the mixing matrix
    (A_ev) and its derivatives 
    """


def invN_from_pysm_instrument(pysm_instrument):
    return invN


def A_ev_invN_from_pysm_instrument_analytic_components(pysm_instrument,
                                                       *analytic_components):
    return A_ev, invN
