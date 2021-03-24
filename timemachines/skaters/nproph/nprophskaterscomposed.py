from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory
from timemachines.skatertools.composition.residualcomposition import residual_chaser_factory
from timemachines.skaters.nproph.nprophskaterssingular import fbnprophet_univariate, fbnprophet_exogenous, fbnprophet_cautious


def fbnprophet_exogenous_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Chase residuals, somewhat cautiously using, quickly moving average """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=fbnprophet_exogenous)


def fbnprophet_univariate_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Chase residuals, somewhat cautiously using, quickly moving average """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=fbnprophet_univariate)


def fbnprophet_cautious_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    """ Chase residuals, somewhat cautiously using, quickly moving average """
    return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=fbnprophet_cautious)


def fbnprophet_exogenous_exogenous(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=fbnprophet_exogenous, f2=fbnprophet_exogenous  )


def fbnprophet_univariate_univariate(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=fbnprophet_univariate, f2=fbnprophet_univariate )


def fbnprophet_univariate_univariate_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
    return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=fbnprophet_univariate, f2=fbnprophet_univariate_hypocratic )


NPROPHET_SKATERS_COMPOSED = [fbnprophet_exogenous_hypocratic, fbnprophet_univariate_hypocratic, fbnprophet_cautious_hypocratic,
                            fbnprophet_exogenous_exogenous, fbnprophet_univariate_univariate_hypocratic]




if __name__ == '__main__':
    from timemachines.skatertools.data.real import hospital_with_exog
    from timemachines.skatertools.visualization.priorplot import prior_plot
    import matplotlib.pyplot as plt
    k = 1
    y, a = hospital_with_exog(k=k, n=450, offset=True)
    f = fbnprophet_univariate_hypocratic
    err2 = prior_plot(f=f, k=k, y=y, n=450, n_plot=50)
    print(err2)
    plt.show()
    pass
