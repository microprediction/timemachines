from timemachines.optimizers.eloratings import optimizer_population_elo_update, random_optimizer_game
from timemachines.optimizers.alloptimizers import OPTIMIZERS
from timemachines.objectives.allobjectives import CLASSIC_OBJECTIVES


def test_elo_optim():
    N_DIM_CHOICES = [3]
    N_TRIALS_CHOICES = [8]

    elo = {}
    for _ in range(2):
        game_result = random_optimizer_game(optimizers=OPTIMIZERS, objectives=CLASSIC_OBJECTIVES,
                                        n_dim_choices=N_DIM_CHOICES, n_trials_choices=N_TRIALS_CHOICES, tol=0.001)
        elo = optimizer_population_elo_update(optimizers=OPTIMIZERS, elo=elo, game_result=game_result)


if __name__=='__main__':
    test_elo_optim()