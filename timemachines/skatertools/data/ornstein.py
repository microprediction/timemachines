import numpy as np
from timemachines.skatertools.utilities.suppression import no_stdout_stderr
import random


def pull_towards_zero(xs, kappa=0.1):
    """ Take a sequence xs and apply mean reversion """
    dxs = np.diff(xs)
    xt = 0
    ou = [ xs[0] ]
    for dx in dxs:
        xt = xt + dx
        xt = xt - kappa*xt
        ou.append(xt)
    return np.array(ou)


def simulate_arima_like_path(seq_len, reverse=False):
    """  Just a way to generate a time-series that isn't completely trivial in structure
    :param seq_len:
    :param plot:
    :return:
    """
    import pandas as pd
    import statsmodels.api as sm

    # Load data
    y = sm.datasets.macrodata.load_pandas().data['cpi']
    for j in range(len(y)):
        y[j] += np.random.randn()

    y.index = pd.period_range('1959Q1', '2009Q3', freq='Q')

    # Create and fit a model once to generate data
    p = random.choice(range(2, 20))
    d = random.choice(range(0, 3))
    q = random.choice(range(0, 3))
    scale = random.choice([2, 1, 0.5, -0.5, -1, -2])
    with no_stdout_stderr():
        mod = sm.tsa.SARIMAX(y, order=(p, d, q), trend='c')
        res = mod.fit()
        res.model_orders['ar_true'] = p
        res.model_orders['ma_true'] = q
    from pprint import pprint
    pprint(res.model_orders)

    with no_stdout_stderr():
        sim = res.simulate(seq_len, anchor='end', repetitions=1)

    # Make sure the data doesn't go crazy in scale
    x = sim.reset_index()['cpi'].values.squeeze() * scale
    x = (x - x[-1]) / 10.0
    if reverse:
        ou = np.flip(pull_towards_zero(np.flip(x)))
    else:
        ou = pull_towards_zero(x)

    return ou


from timemachines.inclusion.matplotlibinclusion import using_matplotlib

if using_matplotlib:
    def show_example_ornstein():
        xs = np.cumsum(np.random.randn(500))
        y = pull_towards_zero(xs)
        import matplotlib.pyplot as plt
        t = list(range(len(y)))
        plt.plot(t,xs,t,y)
        plt.grid()
        plt.legend(['raw','ou'])
        plt.show()


    def show_example_arima_like():
        y = simulate_arima_like_path(seq_len=500)
        import matplotlib.pyplot as plt
        t = list(range(len(y)))
        plt.plot(t,y)
        plt.grid()
        plt.legend(['raw','ou'])
        plt.show()
else:
    def show_example_ornstein():
        print('pip install matplotlib')

    def show_example_arima_like():
        print('pip install matplotlib')


if __name__=='__main__':
    show_example_arima_like()



