
from timemachines.skaters.ik.ikinclusion import using_ik

if using_ik:
    from timemachines.skaters.ik.ikwrappers import ik_nnma3_iskater, ik_nnma10_iskater, ik_nnma100_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def ik_nn_ma3(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = float, e: E_TYPE = None, r: R_TYPE = float, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=ik_nnma3_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    def ik_nn_ma10(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = float, e: E_TYPE = None, r: R_TYPE = float, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=ik_nnma10_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    def ik_nn_ma100(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = float, e: E_TYPE = None, r: R_TYPE = float, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=ik_nnma100_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    IK_NN_MA_SKATERS = [ik_nn_ma3, ik_nn_ma10, ik_nn_ma100]

else:
    IK_NN_MA_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    err1 = hospital_mean_square_error_with_sporadic_fit(f=ik_nnma3_iskater, k=3, n=110)