from timemachines.skatertools.tuning.hyper import using_humpday
if using_humpday:
    from humpday.optimizers.alloptimizers import OPTIMIZERS
    from humpday.objectives.classic import CLASSIC_OBJECTIVES
    import random

    # Just to check if someone (ahem) breaks humpday


    def test_optimizers():
        n_trials = 10
        n_dim = random.choice([2,3])
        optimizer = random.choice(OPTIMIZERS)
        objective = random.choice(CLASSIC_OBJECTIVES)
        best_val, best_x, feval_count = optimizer(objective, n_trials=n_trials,n_dim=n_dim,with_count=True)


if __name__=='__main__':
    test_optimizers()