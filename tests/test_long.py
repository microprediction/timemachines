
from timemachines.inclusion.pandasinclusion import using_pandas

if using_pandas:

    def test_long():
        from timemachines.skatertools.data.long import random_long_data
        t,y = random_long_data(n_obs=30000)
        assert len(y)==30000


if __name__=='__main__':
    test_long()