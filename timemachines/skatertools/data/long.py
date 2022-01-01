import time
import random

URL_TEMPLATE = 'https://raw.githubusercontent.com/microprediction/precisedata/main/returns/fathom_data_N.csv'

n_data = 450

from timemachines.inclusion.pandasinclusion import using_pandas

if using_pandas:
    import pandas as pd

    def random_long_data(n_obs):
        """ Returns random long time series with reasonable amount of signal """
        assert n_obs<35000

        start_time = time.time()
        t = [ start_time+24*60*60*60*i for i in range(n_obs) ]

        got = False
        while not got:
            the_choice = random.choice(list(range(n_data)))
            the_url = URL_TEMPLATE.replace('N', str(the_choice))
            try:
                df = pd.read_csv(the_url)
                del df['Unnamed: 0']
                got = len(df.index) > n_obs+10
            except:
                got = False
        col = random.choice(list(df.columns))
        vals = list(df[col].values)
        k = random.choice(list(range(0,len(df.index)-n_obs-5)))
        y = vals[k:k+n_obs]
        return y,t


if __name__=='__main__':
    y,t = random_long_data(n_obs=1000)
    print(y[:10])