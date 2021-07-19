from timemachines.skaters.divine.divineinclusion import using_divinity, dv
if using_divinity:
    from timemachines.skaters.divine.divineskaters import divine_univariate
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit


    def dont_test_divine(): # too noisy
        err = hospital_mean_square_error_with_sporadic_fit(f=divine_univariate, n=105) # Won't get past warmup so not a real test


if __name__=='__main__':
    assert using_divinity,'pip install divinity'
    dont_test_divine()