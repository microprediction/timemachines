from timemachines.skatertools.offlinetesting.optimizerandomskater import optimize_random_skater
import random

REGRESSION_TESTS = [optimize_random_skater]





if __name__=='__main__':
     task = random.choice(REGRESSION_TESTS)
     print(task.__name__)
     task()