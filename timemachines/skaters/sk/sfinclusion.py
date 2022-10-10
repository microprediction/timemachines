
try:
    from statsforecast.core import StatsForecast
    from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
    using_statsforecast = True
except ImportError:
    using_statsforecast = False

if __name__=='__main__':
    print({'using_statsforecast':using_statsforecast})
    from sktime.forecasting.statsforecast import StatsForecastAutoARIMA