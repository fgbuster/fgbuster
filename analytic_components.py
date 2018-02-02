""" Components that have an analytic frequency emission law

These classes have to provide the following functionality:
    - are constructed from an arbitrary analytic expression
    - provide an efficient evaluator that takes as argument frequencies and
      parameters
    - same thing for the gradient wrt the parameters

For frequent components (e.g. power law, gray body) these classe are already
prepared.

(see hpcs.components)
"""
