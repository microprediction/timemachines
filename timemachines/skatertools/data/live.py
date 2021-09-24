try:
    from microprediction import MicroReader
    using_micro = True
except ImportError:
    using_micro = False

import random
import time

# Live data streams at www.microprediction.com

if using_micro:
    import numpy as np

    def random_stream_name_url_lagged(min_len=500, exclude_str=None, include_str=None,
                                      with_lagged=False,
                                      with_url=True,
                                      backoff_seconds=1):
        """ Randomly selected univariate series
                :returns:  name, url, lagged if with_lagged=True
        """
        mr = MicroReader()
        names = mr.get_stream_names()
        okay = False
        while not okay:
            name = random.choice(names)
            okay = True
            if exclude_str is not None and exclude_str in name:
                okay = False
            if include_str is not None and include_str not in name:
                okay = False
            if okay and min_len>0:
                lagged = mr.get_lagged(name=name, count=10000)
                n_obs = len(lagged)
                okay = n_obs >= min_len
                if not okay:
                    time.sleep(backoff_seconds)
        url = 'https://www.microprediction.org/stream_dashboard.html?stream=' + name.replace('.json', '')
        if with_lagged and with_url:
            return name, url, lagged
        elif with_url and (not with_lagged):
            return name, url
        elif (not with_url) and (with_lagged):
            return name, lagged
        elif (not with_url) and (not with_lagged):
            return name


    def random_regular_stream_name(min_len=0, with_url=False):
        """ Randomly selected univariate series with some data  """
        return random_stream_name_url_lagged(min_len=min_len, exclude_str='~',with_url=with_url, with_lagged=False)


    def random_chronological_values_and_times(min_len=500, exclude_str=None, include_str=None, backoff_seconds=1):
       _name, _url, lagged = random_stream_name_url_lagged(min_len=min_len, exclude_str=exclude_str, include_str=include_str,
                                                         with_lagged=True, backoff_seconds=backoff_seconds)
       # Chronological
       lagged_values = [l[1] for l in reversed(lagged)]
       lagged_times = [l[0] for l in reversed(lagged)]
       return lagged_values, lagged_times

    random_stream_data = random_chronological_values_and_times

    def stream_data(name:str,n_obs:int):
        """ values and times for a univariate stream """
        mr = MicroReader()
        lagged_values, lagged_times = mr.get_lagged_values_and_times(name=name, count=n_obs)
        y, t = list(reversed(lagged_values)), list(reversed(lagged_times))
        return y, t


    def random_regular_data(n_obs=500):
        """ Retrieve univariate time series chosen randomly from the collection at www.microprediction.org
            Chronological
        :returns:  y, t
        """
        y,t = random_chronological_values_and_times(min_len=n_obs, exclude_str='~')
        return y[:n_obs],t[:n_obs]


    def random_residual_data(n_obs=500):
        """ Retrieve univariate time series chosen randomly from the collection at www.microprediction.org
        :returns:  y, t
        """
        y, t = random_chronological_values_and_times(min_len=n_obs, include_str='~')
        return y[:n_obs], t[:n_obs]


    def random_rescaled_chronological_values_and_times(n_obs=900, include_str='electricity', exclude_str='~'):
        y, t = random_chronological_values_and_times(min_len=n_obs, include_str=include_str, exclude_str=exclude_str)
        scale = (max(y) -min(y) )+0.001
        return [ yt/scale for yt in y[:n_obs] ], t[:n_obs]


    def random_surrogate_data(f, k=1, n_real=50, n_samples=200, n_warm = 500, n_input=20, verbose=False,
                               include_str='electricity',exclude_str='~'):
        """ Create dataset of skater input (abbreviated to last n_input lags) and output
        :param f:   skater
        :param k:   step ahead
        :param n_real:   number of distinct real time series to use
        :param n_input:  number of lags to use in training
        :param n_warm:   number of data points to use to warm up skater before harvesting predictions
        :return: x_train, y_train, y_true
        """
        n_obs = n_samples + n_warm + 2* k + 10
        n_train = n_real * n_samples
        x_train = np.zeros(shape=(n_train, n_input, 1))
        y_train = np.zeros(shape=(n_train, 1))
        x_stds = [1.0]
        y_true = list()
        if verbose:
            print(np.shape(x_train))
        for ndx_real in range(n_real):
            # Get real data
            y_real, _ = random_rescaled_chronological_values_and_times(n_obs=n_obs, include_str=include_str, exclude_str=exclude_str)
            if verbose:
                print(y_real[:5])
                print((ndx_real, n_real))
                print(np.mean(x_stds))
            # Warm the model up
            y_real_warm = y_real[:n_warm]
            s = {}
            for y in y_real_warm:
                x, x_std, s = f(y, k=k, s=s)
            # Now create examples of k-step ahead forecasts
            y_real_harvest = y_real[n_warm:n_warm + n_samples]
            for j2, y in enumerate(y_real_harvest):
                x, x_std, s = f(y, k=k, s=s)
                xk = x[k - 1]
                j = ndx_real * n_samples + j2
                yj_true = y_real[n_warm + j2 + k]
                y_true.append(yj_true)
                y_train[j, 0] = xk
                x_stds.append(x_std[-k])
                reverse_input_data = y_real[n_warm + j2 - n_input+1:n_warm +j2+1]
                input_data = reverse_input_data
                #assert input_data[0]==y
                for l in range(n_input):
                    x_train[j, l, 0] = input_data[l]

        return x_train, y_train, y_true


if __name__=='__main__':
    if using_micro:
        name, url = random_regular_stream_name(min_len=900,with_url=True)
        print(url)