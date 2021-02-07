from timemachines.skaters.dlm.dlmunivariate import dlm_univariate_r3, dlm_univariate_a, dlm_univariate_b
from timemachines.skaters.dlm.dlmexogenous import dlm_exogenous_a, dlm_exogenous_b, dlm_exogenous_r3

DLM_SKATERS = [ dlm_univariate_a, dlm_univariate_b ]
DLM_R3_SKATERS = [ dlm_univariate_r3, dlm_exogenous_r3 ]