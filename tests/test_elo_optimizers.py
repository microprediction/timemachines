from timemachines.optimizers.eloratings import optimizer_population_elo_update


def test_elo_optim():
    elo = {}
    elo, _ = optimizer_population_elo_update(elo=elo)
    elo, _ = optimizer_population_elo_update(elo=elo)
