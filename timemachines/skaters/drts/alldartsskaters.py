
from timemachines.skaters.drts.dartsarimaskater import DARTS_ARIMA_SKATERS
from timemachines.skaters.drts.dartsautoarimaskater import DARTS_AUTOARIMA_SKATERS
from timemachines.skaters.drts.dartsexpsmoothingskater import DARTS_EXPONENTIALSMOOTHING_SKATERS
from timemachines.skaters.drts.dartsfftskater import DARTS_FFT_SKATERS
from timemachines.skaters.drts.dartsfourthetaskater import DARTS_FOURTHETA_SKATERS
from timemachines.skaters.drts.dartsprophetskater import DARTS_PROPHET_SKATERS
from timemachines.skaters.drts.dartsthetaskater import DARTS_THETA_SKATERS
from timemachines.skaters.drts.dartsnbeatsskater import DARTS_NBEATS_SKATERS

DARTS_SKATERS = DARTS_ARIMA_SKATERS + DARTS_AUTOARIMA_SKATERS + DARTS_EXPONENTIALSMOOTHING_SKATERS + DARTS_FFT_SKATERS + DARTS_FOURTHETA_SKATERS + DARTS_PROPHET_SKATERS + DARTS_THETA_SKATERS + DARTS_NBEATS_SKATERS

# Took out nbeats for now because it takes up a huge amount of disk space without asking nicely 
DARTS_SKATERS = DARTS_ARIMA_SKATERS + DARTS_AUTOARIMA_SKATERS + DARTS_EXPONENTIALSMOOTHING_SKATERS + DARTS_FFT_SKATERS + DARTS_FOURTHETA_SKATERS + DARTS_PROPHET_SKATERS + DARTS_THETA_SKATERS
