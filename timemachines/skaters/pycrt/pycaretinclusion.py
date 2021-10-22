try:
    from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
    using_pycaret=True
except ImportError:
    using_pycaret=False
