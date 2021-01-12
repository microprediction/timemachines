import matplotlib.pyplot as plt
import numpy as np
from timemachines.synthetic import brownian_with_exogenous, brownian_with_noise
from timemachines.skating import prior


# Quick ways to inspect skaters


def prior_plot(f, ys=None, k=None, ats=None, ts=None, e=None, r=0.5, x0=np.nan, n=150, n_plot=25):
    """
         Apply state machine to univariate series,
         Show observations, and out of sample predictions predictions
    """
    if ys is None:
       ys = brownian_with_noise(n=n)

    if ts is None:
        ts = range(len(ys))

    xs = prior(f=f,ys=ys, k=k, ats=ats, ts=ts, e=e, r=r, x0=x0 )
    ysf = [ [y_] for y_ in ys ]
    plot_observations_and_predictions(ts=ts, xs=xs, ys=ysf, n_plot=n_plot)


def prior_plot_exogenous(f, ys=None, k=None, ats=None, ts=None, e=None, r=0.5, x0=np.nan, n=150, n_plot=25):
    """
          Apply state machine to univariate series,
          Show observations, out of sample predictions predictions, and exogenous variables
     """
    if ys is None:
       ys = brownian_with_exogenous(n)

    if ts is None:
       ts = range(len(ys))

    xs = prior(f=f, ys=ys, k=k, ats=ats, ts=ts, e=e, r=r, x0=x0)
    plot_observations_and_predictions(ts=ts, xs=xs, ys=ys, n_plot=n_plot)


def plot_observations_and_predictions(ts, xs, ys, n_plot:int):
    """
    :param ts:        Time of observation
    :param xs:        Point estimates
    :param ys:        Univariate or Multivariate series of observations
    :param n_plot:    Number of examples to plot (from end)
    :return:
    """
    try:
        ys0 = [y_[0] for y_ in ys]
    except:
        ys0 = ys
    plt.plot(ts[-n_plot:], ys0[-n_plot:], 'b*')
    lgnd = ['target']
    for j in range(1,len(ys[0])):
        ysj = [y_[j] for y_ in ys]
        plt.plot(ts[-n_plot:], ysj[-n_plot:], 'r+')
        lgnd.append('exogenous '+str(j))

    # Run the model
    plt.plot(ts[-n_plot:], xs[-n_plot:], 'g-')
    lgnd.append('prediction')
    plt.legend(lgnd)
    plt.show()


