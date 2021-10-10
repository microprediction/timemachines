try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    using_sklearn = True
except ImportError:
    using_sklearn = False
