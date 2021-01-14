from timemachines.stochastictests.allstochastictests import STOCHASTIC_TESTS
import time
import random
TIMEOUT = 60*5

# Regression tests run occasionally to check various parts of hyper-param spaces, etc.

if __name__=='__main__':
    start_time = time.time()
    elapsed = time.time()-start_time
    while elapsed < TIMEOUT:
        a_test = random.choice(STOCHASTIC_TESTS)
        print('Running '+str(a_test.__name__))
        a_test()
        elapsed = time.time() - start_time
