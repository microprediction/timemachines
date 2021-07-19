from timemachines.skaters.tsa.tsaconstant import tsa_p1_d0_q0, tsa_aggressive_ensemble
from timemachines.skaters.tsa.tsahypocratic import quickly_hypocratic_tsa_d0
from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit

TSA_TO_TEST = [ tsa_p1_d0_q0, tsa_aggressive_ensemble, quickly_hypocratic_tsa_d0 ]


def test_tsa():
    for f in TSA_TO_TEST:
        err = hospital_mean_square_error_with_sporadic_fit(f=f, k=5, n=50, fit_frequency=1, n_test=10)



if __name__=='__main__':
    test_tsa()