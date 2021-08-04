
from timemachines.skaters.linkedingreykite.linkedingreykiteinclusion import using_greykite

if using_greykite:
    from timemachines.skaters.linkedingreykite.linkedingreywrappers import linkedin_greykite_iskater
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.batch.batchskater import batch_skater_factory

    def linkedin_greykite_basic_skater(y: Y_TYPE, s, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None, r: R_TYPE = None, emp_mass=0.0):
        return batch_skater_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=r, emp_mass=emp_mass,
                                    iskater=linkedin_greykite_iskater,
                                    iskater_kwargs={},
                                    min_e=0)

    LINKEDINGREYKITE_BASIC_SKATERS = [linkedin_greykite_basic_skater]

else:
    LINKEDINGREYKITE_BASIC_SKATERS = []

if __name__=='__main__':
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
    err1 = hospital_mean_square_error_with_sporadic_fit(f=linkedin_greykite_basic_skater, k=3, n=110)