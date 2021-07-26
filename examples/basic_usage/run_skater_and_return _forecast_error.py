from timemachines.skaters.simple.thinking import thinking_slow_and_slow
from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit


if __name__=='__main__':
    print(hospital_mean_square_error_with_sporadic_fit(f=thinking_slow_and_slow,n=120,fit_frequency=10))