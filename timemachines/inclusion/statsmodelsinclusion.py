try:
   from statsmodels.tsa.arima.model import ARIMA
   using_statsmodels = True
except ImportError:
    try:
       from statsmodels.tsa.arima_model import ARIMA
       using_statsmodels = True
    except:
       using_statsmodels = False
