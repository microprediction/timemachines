try:
    from microprediction import MicroReader
    using_micro = True
except ImportError:
    using_micro = False

import random

# Live data streams at www.microprediction.com

if using_micro:
    import numpy as np

    def random_stream_name(min_len=500,exclude_str=None, include_str=None):
        """ Randomly selected univariate series
                :return:  y, t, url
                """
        mr = MicroReader()
        names = mr.get_stream_names()
        okay = False
        while not okay:
            name = random.choice(names)
            n_obs = len(mr.get_lagged_values(name=name, count=10000))
            okay = True
            if exclude_str is not None and exclude_str in name:
                okay = False
            if include_str is not None and include_str not in name:
                okay = False
            if n_obs < min_len:
                okay = False
        url = 'https://www.microprediction.org/stream_dashboard.html?stream=' + name.replace('.json', '')
        return name, url

    def random_regular(min_len=500):
        """ Randomly selected univariate series
        :return:  y, t, url
        """
        return random_stream_name(min_len=min_len,exclude_str='~')


    def random_stream_data(min_len=500,exclude_str='~', include_str='electricity'):
        """ Randomly selected univariate series
        :return:  y, t, url
        """
        name, url = random_stream_name(min_len=min_len, exclude_str=exclude_str, include_str=include_str)
        return stream_data(name=name, n_obs=1000)


    def random_residual(min_len=500):
        """ Randomly selected univariate series
        :return:  y, t, url
        """
        return random_stream_name(min_len=min_len, include_str='z1~')


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


    def random_residual_data(n_obs=500):
        """ Retrieve univariate time series chosen randomly from the collection at www.microprediction.org
        :returns:  y, t
        """
        name, url = random_residual(min_len=n_obs)
        return stream_data(name=name, n_obs=n_obs)


    def random_rescaled_stream_data(n_obs=900,include_str='electricity',exclude_str='~'):
        y, t = random_stream_data(min_len=n_obs, include_str=include_str,exclude_str=exclude_str)
        dy = np.diff(y)
        scale = (max(np.abs(dy) ) -min(np.abs(dy)) ) +0.001
        return [ yt /scale for yt in y[:n_obs] ]


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
            y_real = random_rescaled_stream_data(n_obs=n_obs, include_str=include_str, exclude_str=exclude_str)
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
                input_data = list(reversed(reverse_input_data))
                assert input_data[-1]==y
                for l in range(n_input):
                    x_train[j, l, 0] = input_data[l]

        return x_train, y_train, y_true


if __name__=='__main__':
    if using_micro:
        name, url = random_regular(min_len=900)
        print(url)