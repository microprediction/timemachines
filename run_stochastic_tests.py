from stochastictests.run_dlm import run_dlm_seasonal_univariate_with_random_hyperparameters
import random
import time
TIMEOUT = 60*5

# Regression tests run occasionally to check various parts of hyper-param spaces, etc.

STOCH_TESTS = [run_dlm_seasonal_univariate_with_random_hyperparameters]


if __name__=='__main__':
    start_time = time.time()
    elapsed = time.time()-start_time
    while elapsed < TIMEOUT:
        a_test = random.choice(STOCH_TESTS)
        print('Running '+str(a_test.__name__))
        a_test()
        elapsed = time.time() - start_time
