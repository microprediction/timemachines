from timemachines.skatertools.data.skaterresiduals import random_multivariate_residual
import numpy as np


if __name__=='__main__':
    X = random_multivariate_residual(n_obs=1000)
    print(np.shape(X))
