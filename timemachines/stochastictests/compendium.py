from timemachines.stochastictests.optimizedlm import optimize_dlm
from timemachines.stochastictests.optimizerandomskater import optimize_random_skater
from timemachines.stochastictests.rundlm import run_dlm

STOCHASTIC_TESTS = [ run_dlm, optimize_dlm, optimize_random_skater ]