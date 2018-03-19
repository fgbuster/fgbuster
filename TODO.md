# Code Requirements
- Map based component separation functionality (pixel domain, but user can define fancy domains)
- Forcasting a la Errard et al 2011
- Forecasting a la Stompor et al 2016
- Take arbitrary models for the reconstruction
- Handy construction of models from analytic expressions
- No hard coded units: the user provides consistent inputs
- Allow 1/f noise
- Allow noise correlated between frequencies (dense weights? diagonalize?)
- Multiple independent patches
- Power spectrum likelihood
- CosmoMC plugin for arbitrary cosmological parameters?
- delensing
- visualization library

# TO DO items
2 - write notebook for users of the library
	2.1 creation of the sky / instrument / beta estimation / cleaning / residuals / cosmo analysis 
3 - write the LiteBIRD code with multipatch and xForecast capability
	3.1. work on the ``smart'' multipatch approach
