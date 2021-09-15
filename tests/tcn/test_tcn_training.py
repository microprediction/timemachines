
from timemachines.skaters.tcn.tcninclusiontraining import using_tcntraining

if using_tcntraining:
    from timemachines.skaters.elo.eloensembles import elo_fastest_residual_precision_ensemble
    from timemachines.skaters.tcn.tcntraining import train_tcn_surrogate

    def test_tcn_training():
        f = elo_fastest_residual_precision_ensemble
        n_lags = 20
        onnx_models = train_tcn_surrogate(f=f, k=1, n_real=1, n_samples=40, n_warm=50, n_tile=2,
                                          n_input=n_lags, verbose=True, n_iterations=100, n_models=3)

