from timemachines.skaters.pmd import pmd_auto
from timemachines.skaters.dlmseasonal import dlm_seasonal
from timemachines.skaters.dlmauto import dlm_auto
from timemachines.skaters.dlmexog import dlm_exog

ALL_SKATERS = [ pmd_auto, dlm_auto, dlm_seasonal, dlm_exog ]
EXOG_SKATERS = [ f for f in ALL_SKATERS if 'exog' in f.__name__ ]
SKATERS = [ f for f in ALL_SKATERS if 'exog' not in f.__name__ ]