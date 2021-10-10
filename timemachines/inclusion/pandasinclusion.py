try:
    import pandas as pd
    using_pandas = True
except ImportError:
    pd = 'pip install pandas'
    using_pandas = False
