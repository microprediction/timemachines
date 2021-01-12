from timemachines.conventions import to_space
from microconventions.zcurve_conventions import ZCurveConventions
from timemachines.evaluation import evaluate_energy, evaluate_mean_squared_error
from timemachines.optimizers.compendium import OPTIMIZERS





def optimize(f, ys, evaluator, optimizer, **kwargs):
    """
    :param f:      skater
    :param ys:     time series
    :param r_dim:  dimensionality to assume for hyper-parameters
    :return:
    """

    def objective(u:[float]):
        """
        :param u:  point in cube
        :return:
        """
        r = ZCurveConventions().from_cube(prctls=list(reversed(u)))
        return evaluate_mean_squared_error(f=f,ys=ys,r=r,**kwargs)




