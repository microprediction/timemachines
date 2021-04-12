from timemachines.skatertools.data.real import hospital
from timemachines.skaters.nproph.nprophetiskaterfactory import using_neuralprophet
if using_neuralprophet:
    from timemachines.skaters.nproph.nprophetiskaterfactory import nprophet_fit_and_predict_simple, nprophet_iskater_factory

    def test_univariate_without_time(show=False):
        k = 3
        n = 100
        y = hospital(n=n)
        x, x_std, forecast, m = nprophet_iskater_factory(y=y, k=k)
        assert len(x) == k
        x1, x_std1, forecast1, m1 = nprophet_fit_and_predict_simple(y=y, k=k)
        if show:
            m.plot(forecast)
            m1.plot(forecast1)
            import matplotlib.pyplot as plt
            plt.show()


