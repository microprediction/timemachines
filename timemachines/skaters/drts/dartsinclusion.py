
try:
    import darts
    from darts.models import FFT, Prophet, ARIMA, AutoARIMA, ExponentialSmoothing, Theta, FourTheta, TransformerModel, \
        NBEATSModel
    using_darts = True
except (ImportError, OSError):
    using_darts = False

