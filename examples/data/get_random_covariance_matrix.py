from timemachines.skatertools.data.skaterresiduals import random_multivariate_residual
from pprint import pprint

if __name__=='__main__':
    df = random_multivariate_residual(n_obs=1000)
    the_cov = df.cov()
    pprint(the_cov[:3].transpose())
