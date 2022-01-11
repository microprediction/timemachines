import time
import random

SKATER_RESIDUAL_URL = 'https://raw.githubusercontent.com/microprediction/precisedata/main/skaterresiduals/skater_residuals_0.csv'

n_data = 450

from timemachines.inclusion.pandasinclusion import using_pandas

if using_pandas:
    import pandas as pd

    def random_skater_residual_dataframe(n_obs:int):
        got = False
        while not got:
            the_choice = random.choice(list(range(n_data)))
            the_url = SKATER_RESIDUAL_URL.replace('N', str(the_choice))
            try:
                df = pd.read_csv(the_url)
                del df['Unnamed: 0']
                got = len(df.index) > n_obs + 10
            except:
                got = False
        return df

    def random_long_residual(n_obs):
        """ Returns random long time series of skater residuals """
        assert n_obs<35000

        start_time = time.time()
        t = [ start_time+24*60*60*60*i for i in range(n_obs) ]
        df = random_skater_residual_dataframe(n_obs)
        col = random.choice(list(df.columns))
        vals = list(df[col].values)
        k = random.choice(list(range(0,len(df.index)-n_obs-5)))
        y = vals[k:k+n_obs]
        return y,t

    def random_multivariate_residual(n_obs, as_dataframe=True, random_start=True):
        """ Canned skater residual data, potentially useful for studying ensembles """
        assert n_obs < 35000
        start_time = time.time()
        t = [start_time + 24 * 60 * 60 * 60 * i for i in range(n_obs)]
        df = random_skater_residual_dataframe(n_obs=n_obs)
        if random_start:
            k = random.choice(list(range(0, len(df.index) - n_obs - 5)))
            df = df[k:k+n_obs]
        return df[:n_obs] if as_dataframe else df[:n_obs].values


if __name__=='__main__':
    y,t = random_long_residual(n_obs=1000)
    print(y[:10])