
try:
    import merlion
    using_merlion = True
except ImportError:
    using_merlion = False
    
try:
    from merlion.models.forecast.prophet import Prophet, ProphetConfig
    using_merlion_prophet = True
except ImportError:
    using_merlion_prophet = False

if __name__=='__main__':
    print(using_merlion)
    print(using_merlion_prophet)
