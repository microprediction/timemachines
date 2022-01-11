from timemachines.skatertools.data.skaterresiduals import random_multivariate_residual


if __name__=='__main__':
    df = random_multivariate_residual(n_obs=1000)
    the_corrcoef = df.corr()
    print(the_corrcoef[:3].transpose())
