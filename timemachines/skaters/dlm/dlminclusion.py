
try:
    from pydlm import dlm, trend, seasonality, dynamic, autoReg
    using_dlm = True
except ImportError:
    class Nothing:
        yup = False
    using_dlm = False
    dlm=Nothing()
    trend=Nothing()
    seasonality=Nothing()
    dynamic =Nothing()
    autoReg=Nothing()