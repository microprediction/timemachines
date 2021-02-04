from timemachines.skaters.divine.divineskaterfactory import divinity_univariate_factory


def divine_univariate(y,s,k,a=None,t=None,e=None):
    return divinity_univariate_factory(y=y, s=s, k=k, a=a, t=t, e=e)


DIVINE_SKATERS = [ divine_univariate ]