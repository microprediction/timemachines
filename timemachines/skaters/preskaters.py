
# Not yet skaters, since they haven't had r set
# Work in progress to expose and tune en masse

from timemachines.skaters.proph.allprophetskaters import PROPHET_R2_SKATERS
from timemachines.skaters.simple.movingaverage import EMA_R1_SKATERS

SKATERS_R3 = []
SKATERS_R2 = PROPHET_R2_SKATERS
SKATERS_R1 = EMA_R1_SKATERS