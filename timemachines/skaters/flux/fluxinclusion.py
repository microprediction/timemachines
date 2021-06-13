try:
    import pyflux as pf
    using_pyflux = True
except ImportError:
    class Mock:
        ARIMA = None
    pf = Mock()
    print('pyflux is off the list while the package on PyPI is broken. You can install from git')
    print('pip install git+https://github.com/RJT1990/pyflux.git')
    using_pyflux = False