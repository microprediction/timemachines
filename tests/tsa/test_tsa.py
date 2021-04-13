from timemachines.skaters.tsa.tsaconstant import tsa_p1_d0_q0, tsa_aggressive_ensemble
from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error

TSA_TO_TEST = [ tsa_p1_d0_q0, tsa_aggressive_ensemble ]


def test_tsa():
    for f in TSA_TO_TEST:
        err = hospital_mean_square_error(f=f, k=5, n=150)



if __name__=='__main__':
    test_tsa()