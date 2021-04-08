from timemachines.skaters.nproph.allnprophskaters import NPROPH_SKATERS
from timemachines.skaters.nproph.nprophskaters import nproph_exogenous

# TODO: what evaluators will work for nproph implementation?
from timemachines.skatertools.evaluation.evaluators import (
    hospital_mean_square_error, hospital_exog_mean_square_error
)

def test_nproph_auto_univariate():
    for f in NPROPH_SKATERS:
        err = hospital_mean_square_error(f=f, k=5, n=150)

# TODO: more predict strategies go here
    
# def test_pmd_auto_exogenous():
#     err = hospital_exog_mean_square_error(
#         f=pmd_exogenous, k=3, n=150
#     )


if __name__=='__main__':
    test_nproph_auto_univariate()
    
    
    # TODO: more predict strategies go here
    # test_pmd_auto_exogenous()