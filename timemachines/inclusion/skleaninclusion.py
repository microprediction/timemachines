try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    using_sklean = True
except ImportError:
    using_sklean = False
