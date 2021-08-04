from timemachines.skaters.linkedingreykite.linkedingreykiteinclusion import using_greykite

if using_greykite:
    import numpy as np
    import pandas as pd
    from typing import List
    import datetime
    from timemachines.skatertools.utilities.suppression import no_stdout_stderr
    from greykite.framework.templates.autogen.forecast_config import ForecastConfig
    from greykite.framework.templates.autogen.forecast_config import MetadataParam
    from greykite.framework.templates.forecaster import Forecaster 
    from greykite.framework.templates.model_templates import ModelTemplateEnum
    from greykite.framework.utils.result_summary import summarize_grid_search_results

    def linkedin_greykite_iskater(y: [[float]], k: int, a: List = None, t: List = None, e=None, deseasonalize=False):
        """
            Calls Linkedin Greykite's forecasting model, but ignores t if supplied.
        """
        if a:
            assert len(a) == len(y) + k
        if np.isscalar(y[0]):
            y0s = [ yt for yt in y]
        else:
            y0s = [ yt[0] for yt in y ]

        idx = pd.date_range("1970-01-01", periods=len(y0s), freq="H")
        df = pd.DataFrame(y0s, index=idx).reset_index().rename(columns={'index':'ts',0:'y'})

        metadata = MetadataParam(
            time_col="ts",
            value_col="y",
            freq="H"
        )

        forecaster = Forecaster()
        result = forecaster.run_forecast_config(
             df=df,
             config=ForecastConfig(
                 model_template=ModelTemplateEnum.SILVERKITE.name,
                 forecast_horizon=k,
                 coverage=0.68,
                 metadata_param=metadata
             )
         )

        forecast = result.forecast
        x = list(forecast.df[-k:].forecast)
        x_std = list((forecast.df.forecast_upper - forecast.df.forecast_lower)/2)[-k:]
        return x, x_std

if __name__=='__main__':
    assert using_greykite, 'pip install greykite'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = linkedin_greykite_iskater(y=y, k=5)
    print(x)
    print(time.time()-st)