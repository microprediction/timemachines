
from timemachines.inclusion.statsmodelsinclusion import using_statsmodels
from timemachines.inclusion.sklearninclusion import using_sklearn
import numpy as np

if using_statsmodels and using_sklearn:

    from timemachines.skaters.tsa.tsaconstant import tsa_p1_d0_q0, tsa_p1_d0_q1

    fs = [ tsa_p1_d0_q0, tsa_p1_d0_q1 ]

    from timemachines.skatertools.ensembling import precision_weighted_ensemble_factory
    
    def my_skater(y,s,k,**ignore):
        return precision_weighted_ensemble_factory(y=y,s=s,k=k,fs=fs)
   
    s = {}
    for y in np.random.randn(25):
        x, x_std, s = my_skater(y=y,s=s,k=1)


