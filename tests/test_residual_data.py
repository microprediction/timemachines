
from timemachines.inclusion.pandasinclusion import using_pandas
import numpy as np

if using_pandas:

    def test_residual_data():
        from timemachines.skatertools.data.skaterresiduals import random_multivariate_residual
        X = random_multivariate_residual(n_obs=100, random_start=False)
        print(np.shape(X))

    def test_residual_data_random():
        from timemachines.skatertools.data.skaterresiduals import random_multivariate_residual
        X = random_multivariate_residual(n_obs=100, random_start=True)
        print(np.shape(X))


if __name__=='__main__':
    test_residual_data()
    test_residual_data_random()