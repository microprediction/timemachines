import warnings

from timemachines.skatertools.tuning.hyper import using_humpday

try:
    from microprediction import MicroReader
    using_microprediction = True
except ImportError:
    warnings.warn('microprediction is not installed so empirical fit cannot occur. pip install microprediction')
    using_microprediction = False


if using_microprediction and using_humpday:

    from timemachines.skatertools.tuning.hyper import optimal_r

    def optimal_r_for_stream(f, name:str, k: int, evaluator = None, optimizer = None,
                                n_trials = None, n_dim = None,  n_burn:int = None, test_objective_first = True)->(
        float, float, dict):   # best_r, best_val, info
        """  Find the best hyper-parameters for a univariate skater using live from www.microprediction.org
        :param f:
        :param name:     Choose from https://www.microprediction.org/browse_streams.html but add '.json' to the end
        :param k:
        :param evaluator:
        :param optimizer:
        :param n_trials:
        :param n_dim:
        :param n_burn:
        :param test_objective_first:
        :return: best_r, best_value, info
        """
        mr = MicroReader()
        lagged_values, lagged_times = mr.get_lagged_values_and_times(name=name)
        t = list(reversed(lagged_times))
        y = list(reversed(lagged_values))
        return optimal_r(f=f,y=y,k=k, a=None,t=t,e=None,evaluator=evaluator,optimizer=optimizer,n_trials=n_trials,
                          n_dim=n_dim, n_burn=n_burn, test_objective_first=test_objective_first)

