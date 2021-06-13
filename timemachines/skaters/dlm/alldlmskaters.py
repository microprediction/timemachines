from timemachines.skaters.dlm.dlmunivariate import using_dlm
if using_dlm:
    from timemachines.skaters.dlm.dlmunivariate import dlm_univariate_r3, dlm_univariate_a, dlm_univariate_b
    from timemachines.skaters.dlm.dlmexogenous import dlm_exogenous_a, dlm_exogenous_b, dlm_exogenous_r3
    from timemachines.skaters.dlm.dlmcomposed import DLM_SKATERS_COMPOSED

    DLM_SKATERS = [ dlm_univariate_a, dlm_univariate_b ] + DLM_SKATERS_COMPOSED
    DLM_R3_SKATERS = [ dlm_univariate_r3, dlm_exogenous_r3 ]
else:
    DLM_SKATERS = []
    DLM_R3_SKATERS = []