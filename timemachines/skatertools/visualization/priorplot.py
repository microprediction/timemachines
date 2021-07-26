
import numpy as np
from timemachines.skatertools.data.synthetic import brownian_with_exogenous, brownian_with_noise
from timemachines.skating import prior
from timemachines.skatertools.utilities.conventions import wrap

# Quick ways to inspect skaters

try:
    import matplotlib.pyplot as plt
    using_matplotlib = True
except ImportError:
    using_matplotlib = False
    plt = None


def prior_plot(f, y=None, k=1, t=None, e=None, r=None, x0=np.nan, n=150, n_plot=25):
    """
         Apply state machine to univariate series,
         Show observations and out of sample predictions predictions
    """
    assert using_matplotlib, 'pip install matplotlib'

    if y is None:
       y = brownian_with_noise(n=n)

    if t is None:
        t = [float(ti) for ti in range(len(y))]

    x, x_std = prior(f=f, y=y, k=k, a=t, t=t, e=e, r=r, x0=x0)
    ysf = [[wrap(y_)[0]] for y_ in y]
    xk = [xt[-1] for xt in x]
    plot_with_last_value(t=t, x=xk, y=ysf, k=k, n_plot=n_plot)


def prior_plot_exogenous(f, y=None, k=None, a=None, t=None, e=None, r=None, x0=np.nan, n=150, n_plot=25):
    """
          Apply state machine to univariate series,
          Show observations, out of sample predictions predictions, and exogenous variables
    """
    assert using_matplotlib, 'pip install matplotlib'

    if y is None:
       y = brownian_with_exogenous(n)

    if t is None:
       t = range(len(y))

    e = [ None]*(len(y)-n_plot) + [10]*n_plot

    x,x_std = prior(f=f, y=y, k=k, a=a, t=t, e=e, r=r, x0=x0)
    xk = [xt[-1] for xt in x]
    plot_with_last_value(t=t, x=xk, y=y, k=k, n_plot=n_plot)


def plot_with_last_value(t, x, y, k, n_plot:int):
    """
    :param t:        Time of observation
    :param x:        Point estimates
    :param y:        Univariate or Multivariate series of observations
    :param n_plot:    Number of examples to plot (from end)
    :return:
    """
    assert using_matplotlib, 'pip install matplotlib'
    assert isinstance(x[0],float)
    try:
        y0 = [y_[0] for y_ in y]
    except:
        y0 = y
    plt.plot(t[-n_plot:], y0[-n_plot:], 'b*')
    lgnd = ['target']
    for j in range(1, len(y[0])):
        ysj = [y_[j] for y_ in y]
        plt.plot(t[-n_plot:], ysj[-n_plot:], 'r+')
        lgnd.append('exogenous '+str(j))

    lv_x = [y0[0]]*k + y0
    lv_x = lv_x[:len(t)]

    plt.plot(t[-n_plot:], x[-n_plot:], 'g-')
    lgnd.append('prediction')

    plt.plot(t[-n_plot:], lv_x[-n_plot:],'g--')
    lgnd.append('last_value')

    plt.legend(lgnd)
    plt.show()


