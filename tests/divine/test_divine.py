from timemachines.skaters.divine.divineskaters import divine_univariate
from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error


def dont_test_divine(): # too noisy
    err = hospital_mean_square_error(f=divine_univariate, n=105) # Won't get past warmup so not a real test


if __name__=='__main__':
    dont_test_divine()