from timemachines.skaters.simple.thinking import thinking_slow_and_slow
from timemachines.skating import prior
from timemachines.skatertools.data.real import hospital
import math
import numpy as np

if __name__=='__main__':
    y = hospital(n=200)
    x_prior, x_std_prior = prior(f=thinking_slow_and_slow,y=y,k=3)
    x_ahead_k = [ xt[-1] for xt in x_prior ]   # prior returns k-offset vectors
    n_burn = 50
    sq_errors = [ (x1-x2)**2 for x1,x2 in zip(y[n_burn:],x_ahead_k[n_burn:])]
    print(math.sqrt(np.mean(sq_errors)))