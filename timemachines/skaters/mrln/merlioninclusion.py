
try:
    import merlion
    using_merlion = True
except ImportError:
    using_merlion = False
    
try:
    # Urgh... this doesn't succeed in determining that Merlion prophet will actually work.
    # Should perhaps instantiate a little prophet model to check
    import prophet
    from merlion.models.forecast.prophet import Prophet, ProphetConfig
    using_merlion_prophet = True
except ImportError:
    using_merlion_prophet = False

if __name__=='__main__':
    print(using_merlion)
    print(using_merlion_prophet)
