
from timemachines.skaters.orbt.orbitinclusion import using_orbit

if using_orbit:
    from timemachines.skaters.orbt.orbitwrappers import orbit_lgt_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def orbit_lgt_skater_factory(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None,
                 emp_mass=0.0,
                 seasonality=None):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=orbit_lgt_iskater,
                                    iskater_kwargs={'seasonality': seasonality},
                                    min_e=0, n_warm=20)

    def orbit_lgt_12(y,s,k,a=None, t=None,e=None):
        return orbit_lgt_skater_factory(y=y, s=s, k=k, a=a,t=t,e=e, seasonality=12)


    def orbit_lgt_24(y,s,k,a=None, t=None,e=None):
        return orbit_lgt_skater_factory(y, s, k, a=a,t=t,e=e, seasonality=24)

