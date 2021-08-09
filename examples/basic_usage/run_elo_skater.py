from timemachines.skaters.elo.eloensembles import elo_faster_residual_balanced_ensemble
import numpy as np


if __name__=='__main__':
    y = np.cumsum(np.random.randn(1000))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = elo_faster_residual_balanced_ensemble(y=yi, s=s, k=3)
        x.append(xi)
