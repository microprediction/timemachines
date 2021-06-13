from timemachines.skaters.pmd.pmdinclusion import using_pmd, pm
if using_pmd:
    from timemachines.skaters.pmd.allpmdskaters import PMD_SKATERS
    from timemachines.skaters.pmd.pmdskaters import pmd_exogenous
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error, hospital_exog_mean_square_error

    def test_pmd_auto_univariate():
        for f in PMD_SKATERS:
            err = hospital_mean_square_error(f=f, k=5, n=150)


    def test_pmd_auto_exogenous():
        err = hospital_exog_mean_square_error(f=pmd_exogenous, k=3, n=150)


if __name__=='__main__':
    assert using_pmd,'pip install pmdarima'
    test_pmd_auto_univariate()
    test_pmd_auto_exogenous()