from timemachines.skaters.dlm import dlm_seasonal
from timemachines.evaluation import quick_brown_fox_randomized


def run_dlm_seasonal_univariate_with_random_hyperparameters():
    err = quick_brown_fox_randomized(f=dlm_seasonal, n=150)

if __name__=='__main__':
    run_dlm_seasonal_univariate_with_random_hyperparameters()