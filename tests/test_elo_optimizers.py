from timemachines.optimizers.eloratings import optimizer_elo_update


def test_elo_optim():
    elo = {}
    elo, _ = optimizer_elo_update(elo=elo)
    elo, _ = optimizer_elo_update(elo=elo)
