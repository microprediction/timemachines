from timemachines.skaters.divine.divineinclusion import using_divinity
if using_divinity:
    from timemachines.skaters.divine.divineskaterfactory import divinity_univariate_factory
    from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory, slowly_moving_hypocratic_residual_factory

    def divine_univariate(y,s,k,a=None,t=None,e=None):
        return divinity_univariate_factory(y=y, s=s, k=k, a=a, t=t, e=e)


    def divine_univariate_hypocratic_slow(y,s,k,a=None,t=None,e=None):
        return slowly_moving_hypocratic_residual_factory(f=divine_univariate, y=y,s=s,k=k,a=a,t=t,e=e)


    def divine_univariate_hypocratic_fast(y,s,k,a=None,t=None,e=None):
        return quickly_moving_hypocratic_residual_factory(f=divine_univariate, y=y,s=s,k=k,a=a,t=t,e=e)


    DIVINE_SKATERS = [ divine_univariate, divine_univariate_hypocratic_slow, divine_univariate_hypocratic_fast ]
else:
    DIVINE_SKATERS = []