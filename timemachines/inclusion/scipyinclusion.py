try:
    from scipy.stats import energy_distance
    using_scipy = True
except ImportError:
    using_scipy = False
