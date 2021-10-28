
from timemachines.skaters.kts.ktsarimaskater import KATS_PROPHET_SKATERS
from timemachines.skaters.kts.ktsmsesskater import KATS_HOLT_WINTERS_SKATERS
from timemachines.skaters.kts.ktsprophetskater import KATS_QUADRATIC_SKATERS

KATS_SKATERS = KATS_PROPHET_SKATERS + KATS_HOLT_WINTERS_SKATERS + KATS_QUADRATIC_SKATERS
