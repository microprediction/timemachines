from timemachines.skaters.conventions import Y_TYPE, from_space, A_TYPE
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.skaters.evaluation import evaluate_mean_squared_error
import traceback
import time
from microprediction import MicroReader

# Find optimal r for a skater


def optimal_r(f, y:[Y_TYPE],
              k:int, a:[A_TYPE], t=None, e=None,
              evaluator=None, optimizer=None,
              n_trials=None, n_dim=None,  # Which dimension to search in
              n_burn:int=None,
              test_objective_first=True)->(float,float,dict):
    """
          Returns:  best_r: float
                    best_val: float
                    info: {best_rn: interpretation in n_dim space of r,
                           feval_count: number of actual objective evaluations,
                          }

    """
    if evaluator is None:
        evaluator = evaluate_mean_squared_error

    def objective(u:[float]):
        r = from_space(u)
        return evaluator(f=f, y=y, k=k, a=a, t=t, e=r, r=r, n_burn=n_burn)

    if test_objective_first:
        start_time = time.time()
        a_test = objective(u=[0.5]*n_dim)  # Fail fast with easier trace
        elapsed = time.time()-start_time
        print('One evaluation took '+ str(round(elapsed/60,1))+' minutes.')
        print('Come back in '+str(n_trials*elapsed/60)+' minutes.')
    best_val, best_rn, feval_count = optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
    best_r = from_space(best_rn)
    info = {'best_rn':best_rn,'feval_count':feval_count}
    return best_r, best_val, info


def optimal_r_for_stream(f, name:str, k: int, evaluator = None, optimizer = None,
                            n_trials = None, n_dim = None,  n_burn:int = None, test_objective_first = True)->(
    float, float, dict):   # best_r, best_val, info
    """  Find the best hyper-parameters for a univariate skater using data from www.microprediction.org
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






def optimize_proph_against_hospital(k=10, n_extra=100):
    """ Illustrative example

           n_extra is roughly the number of times the prophet model will be called

    """
    from timemachines.skaters.evaluation import evaluate_mean_squared_error
    from timemachines.data.real import hospital_with_exog
    from timemachines.skaters.proph.prophparams import PROPHET_META
    from timemachines.optimizers.optunacube import optuna_tpe_cube
    evaluator = evaluate_mean_squared_error
    n_burn = PROPHET_META['n_warm'] + k + 1
    y, a = hospital_with_exog(k=k, n=n_burn+n_extra, offset=True)

    from timemachines.skaters.proph.prophskaters import fbprophet_univariate, fbprophet_known, fbprophet_recursive
    from timemachines.skaters.simple.basic import BASIC_SKATERS
    skaters = [fbprophet_univariate, fbprophet_known, fbprophet_recursive, fbprophet_recursive]+BASIC_SKATERS

    for f in skaters:
        mn_err = evaluator(f=f, y=y, k=k, a=a, n_burn=n_burn)
        print(f.__name__ + ' : '+ str(mn_err))

    # Finally, try to optimize prophet with hyper-parameters
    from timemachines.skaters.proph.prophskaters import fbprophet_exogenous_r2
    f = fbprophet_exogenous_r2
    n_dim = 2
    best_r, best_val, info = optimal_r(f=f, y=y, k=k, a=a, evaluator=evaluator,
                                 optimizer=optuna_tpe_cube, n_burn=n_burn,
                                 n_trials=50, n_dim=n_dim,
                                 test_objective_first=True)
    from pprint import pprint
    pprint(info)
    print('Best r is ' + str(best_r))
    print('Best value is '+str(best_val))
    from timemachines.skaters.proph.prophparams import prophet_param
    from timemachines.skaters.conventions import to_space
    rn = to_space(best_r)
    pprint({'changepoint prior':prophet_param(param_name='changepoint_prior_scale',u=rn[0]),
           'seasonality prior':prophet_param(param_name='seasonality_prior_scale',u=rn[0])})

    # But out of sample...
    # TODO


def select_working_combinations():
    """ Tedious examination of which optimizers work against different evaluators """
    from timemachines.data.synthetic import brownian_with_exogenous
    from timemachines.skaters.evaluation import EVALUATORS
    broken = list()
    k = 1
    from timemachines.skaters.simple.basic import moving_average_r1
    f = moving_average_r1
    for optimizer in OPTIMIZERS:
        print(' ')
        print(optimizer.__name__)
        for evaluator in EVALUATORS:
            try:
                print(evaluator.__name__, optimal_r(f=f, y=brownian_with_exogenous(n=40), k=k, a=None,
                                                    n_trials=5, n_dim=3, n_burn=20, optimizer=optimizer,
                                                    evaluator=evaluator, with_count=True))
            except Exception as e:
                traceback.print_tb()
                broken.append((optimizer.__name__, evaluator.__name__))
    print(' ')
    print('Broken : ')
    print(broken)


if __name__=='__main__':
    optimize_proph_against_hospital()

