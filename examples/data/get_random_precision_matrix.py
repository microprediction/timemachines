from timemachines.skatertools.data.skaterresiduals import random_multivariate_residual
import numpy as np
import pandas as pd

if __name__=='__main__':
    df = random_multivariate_residual(n_obs=1000)
    the_cov = df.cov()
    df_inv = pd.DataFrame(data=np.linalg.pinv(the_cov.values), columns=the_cov.columns, index=the_cov.index)
    print(df_inv[:3].transpose())
