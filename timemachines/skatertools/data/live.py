try:
    from microprediction import MicroReader
    have_micro = True
except ImportError:
    have_micro = False

import random

# Live data streams at www.microprediction.com

if have_micro:
    def random_regular(min_len=500):
        """ Randomly selected univariate series
        :return:  y, t, url
        """
        mr = MicroReader()
        names = mr.get_stream_names()
        okay = False
        while not okay:
            name = random.choice(names)
            n_obs = len(mr.get_lagged_values(name=name,count=10000))
            okay = n_obs > min_len and '~' not in name
        url = 'https://www.microprediction.org/stream_dashboard.html?stream=' +name.replace('.json','')
        return name, url


    def stream_data(name:str,n_obs:int):
        """ values and times for a univariate stream """
        mr = MicroReader()
        lagged_values, lagged_times = mr.get_lagged_values_and_times(name=name, count=n_obs)
        y, t = list(reversed(lagged_values)), list(reversed(lagged_times))
        return y, t


    def random_regular_data(n_obs=500):
        """ Retrieve univariate time series chosen randomly from the collection at www.microprediction.org
        :returns:  y, t
        """
        name, url = random_regular(min_len=n_obs)
        return stream_data(name=name,n_obs=n_obs)


if __name__=='__main__':
    if have_micro:
        name, url = random_regular(min_len=900)
        print(url)