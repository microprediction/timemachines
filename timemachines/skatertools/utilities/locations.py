PYPI = {'tsa':'statsmodels',
        'fbprophet':'prophet',
        'pmd':'pmdarima',
        'rvr':'river',
        'nprophet':'neuralprophet',
        'dlm':'pydlm',
        'divine':'divinity',
        'orbit':'orbit',
        'bats':'tbats',
        'ats':'auto_ts',
        'glu':'gluonts',
        'flux':'pyflux',
        'sk':'sktime'}


def pypi_from_name(name):
    stem = name.split('_')[0]
    short_name = PYPI.get(stem)
    stub = 'https://pypi.org/project/'
    return stub+short_name if short_name else stub+'timemachines'



if __name__=='__main__':
    from timemachines.skaters.localskaters import LOCAL_SKATERS
    from pprint import pprint
    pprint([(sk.__name__,pypi_from_name(sk.__name__)) for sk in LOCAL_SKATERS])
    print(len(LOCAL_SKATERS))