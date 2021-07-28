from timemachines.skaters.sk.skinclusion import using_sktime
from timemachines.skaters.pmd.pmdinclusion import using_pmd
from timemachines.skatertools.composition.conjugation import exp_conjugation_factory

if using_sktime and using_pmd:
    from timemachines.skaters.sk.skwrappers import sk_autoets_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def sk_ae_factory_for_positive_y(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0,
                      auto=True, trend=None,damped=False):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=sk_autoets_iskater,
                                    iskater_kwargs={'auto':auto,'trend':trend,'damped':damped},
                                    min_e=0, n_warm=20)

    def sk_ae_factory(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0,
                      auto=True, trend=None,damped=False):
        """
           conjugation with exp/log
        """
        f_kwargs = {'emp_mass':emp_mass,'auto':auto, 'trend':trend, 'damped':damped}
        return exp_conjugation_factory(y=y,s=s,k=k, a=a, t=t, e=e, r=r, f=sk_ae_factory_for_positive_y, f_kwargs=f_kwargs)


    def sk_ae(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
                         emp_mass=0.0):
        return sk_ae_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass)


    def sk_ae_add(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
              emp_mass=0.0):
        return sk_ae_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass, trend='add', damped=False)


    def sk_ae_mul(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
                  emp_mass=0.0):
        # Will fail if y<0
        return sk_ae_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass, trend='mul', damped=False)


    def sk_ae_add_damped(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
                  emp_mass=0.0):
        return sk_ae_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass, trend='add', damped=True)


    def sk_ae_mul_damped(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
                         emp_mass=0.0):
        return sk_ae_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass, trend='mul', damped=True)


    SK_AE_SKATERS = [sk_ae, sk_ae_add, sk_ae_add_damped]

else:
    SK_AE_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    for f in SK_AE_SKATERS:
        err1 = hospital_mean_square_error_with_sporadic_fit(f=f, k=3, fit_frequency=70)
