from timemachines.skaters.pmd import pmd_auto
from timemachines.skaters.dlmseasonal import dlm_seasonal
from timemachines.skaters.dlmauto import dlm_auto
from timemachines.skaters.dlmexog import dlm_exog

ALL_SKATERS = [ pmd_auto, dlm_auto, dlm_seasonal, dlm_exog ]
SKATERS = [ dlm_exog ]  # Needing optimization