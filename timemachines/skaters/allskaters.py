from timemachines.skaters.simple.movingaverage import EMA_SKATERS, EMA_R1_SKATERS
from timemachines.skaters.simple.linear import LINEAR_SKATERS
from timemachines.skaters.divine.divineskaters import DIVINE_SKATERS
from timemachines.skaters.proph.allprophetskaters import PROPHET_SKATERS, PROPHET_R2_SKATERS
from timemachines.skaters.dlm.alldlmskaters import DLM_SKATERS
from timemachines.skaters.simple.thinking import THINKING_SKATERS
from timemachines.skaters.simple.hypocraticensemble import HYPOCRATIC_ENSEMBLE_SKATERS
from timemachines.skaters.pmd.allpmdskaters import PMD_SKATERS
from timemachines.skaters.tsa.alltsaskaters import TSA_SKATERS
from timemachines.skaters.nproph.allnprophetskaters import NPROPHET_SKATERS
from timemachines.skaters.uberorbit.allorbitskaters import ORBIT_SKATERS
from timemachines.skaters.bats.allbatsskaters import BATS_SKATERS
from timemachines.skaters.rvr.allriverskaters import RIVER_SKATERS

SKATERS_R3 = []
SKATERS_R2 = PROPHET_R2_SKATERS
SKATERS_R1 = EMA_R1_SKATERS

# Leaving out LINEAR_SKATERS

# And with no hyper-params...
SKATERS = EMA_SKATERS + PROPHET_SKATERS + DIVINE_SKATERS + DLM_SKATERS + \
          THINKING_SKATERS + PMD_SKATERS + TSA_SKATERS + NPROPHET_SKATERS +\
          HYPOCRATIC_ENSEMBLE_SKATERS + ORBIT_SKATERS +BATS_SKATERS + RIVER_SKATERS

FAST_SKATERS = EMA_SKATERS + THINKING_SKATERS + HYPOCRATIC_ENSEMBLE_SKATERS + RIVER_SKATERS


def skater_from_name(name):
    valid = [f for f in SKATERS if f.__name__==name ]
    return valid[0] if len(valid)==1 else None


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
        'flux':'pyflux'}


def pypi_from_name(name):
    stem = name.split('_')[0]
    short_name = PYPI.get(stem)
    stub = 'https://pypi.org/project/'
    return stub+short_name if short_name else stub+'timemachines'


if __name__=='__main__':
    from pprint import pprint
    pprint([(sk.__name__,pypi_from_name(sk.__name__)) for sk in SKATERS])
    print(len(SKATERS))