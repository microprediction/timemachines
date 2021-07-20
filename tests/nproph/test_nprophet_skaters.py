from timemachines.skaters.nproph.nprophetiskaterfactory import using_neuralprophet
if using_neuralprophet:

    from timemachines.skaters.nproph.nprophetskaters import nprophet_p1, nprophet_p8
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit

    NPROPHET_TO_TEST = [ nprophet_p1, nprophet_p8 ]


    def test_ensemble():
        for f in NPROPHET_TO_TEST:
            err = hospital_mean_square_error_with_sporadic_fit(f=f, k=5, n=100, fit_frequency=70)



if __name__=='__main__':
    import time
    st = time.time()
    test_ensemble()
    print(time.time()-st)