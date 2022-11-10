from timemachines.inclusion.sklearninclusion import using_sklearn
from timemachines.skaters.sk.sfinclusion import using_statsforecast
from timemachines.skaters.sk.skinclusion import using_sktime
from timemachines.skaters.pmd.pmdinclusion import using_pmd
from timemachines.skatertools.comparison.comparing import compare
from timemachines.skatertools.data.ornstein import simulate_arima_like_path

# An example of comparing some skaters' performance
# Modify the list of skaters and the data as you see fit


if __name__=='__main__':
    if using_statsforecast and using_sklearn and using_sktime and using_pmd:
        from timemachines.skaters.elo.forever import forever as f1
        from timemachines.skaters.simple.trivial import trivial_last_value as f2
        from timemachines.skaters.sk.sfautoarima import sf_autoarima as f3
        from timemachines.skaters.tsa.tsaconstant import tsa_aggressive_ensemble as f4
        fs = [f1, f2, f3, f4]  # <--- Add as you see fit

        # Make data with different ARIMA-like processes
        ys = []
        for k in range(25):
            ys = ys + list(simulate_arima_like_path(seq_len=500)[100:])

        compare(fs=fs, ys=ys, n_test=10, n_train=200)
    else:
        print('pip install statsforecast')
        print('pip install sktime')
        print('pip install pmdarima')