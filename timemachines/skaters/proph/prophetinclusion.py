try:
    import pandas as pd
    from prophet import Prophet
    using_prophet=True
except ImportError:
    using_prophet=False