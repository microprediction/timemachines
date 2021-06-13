from timemachines.skaters.nproph.nprophetiskaterfactory import using_neuralprophet, NeuralProphet

if using_neuralprophet:
    from timemachines.skaters.nproph.nprophetskaterfactory import nprophet_skater_factory
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
    from typing import List


    def nprophet_p1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                                 t: T_TYPE = None, e: E_TYPE = None):
        return nprophet_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e, recursive = False, model_params = {'n_lags':1})


    def nprophet_p2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                                 t: T_TYPE = None, e: E_TYPE = None):
        return nprophet_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e, recursive = False, model_params = {'n_lags':2})


    def nprophet_p3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                                 t: T_TYPE = None, e: E_TYPE = None):
        return nprophet_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e, recursive = False, model_params = {'n_lags':3})


    def nprophet_p5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                                 t: T_TYPE = None, e: E_TYPE = None):
        return nprophet_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e, recursive = False, model_params = {'n_lags':5})


    def nprophet_p8(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None,
                                 t: T_TYPE = None, e: E_TYPE = None):
        return nprophet_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e, recursive = False, model_params = {'n_lags':5})


    NPROPHET_UNIVARIATE_SKATERS = [ nprophet_p1, nprophet_p2, nprophet_p3, nprophet_p5, nprophet_p8 ]
else:
    NPROPHET_UNIVARIATE_SKATERS = []



