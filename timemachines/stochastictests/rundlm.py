from timemachines.skaters.dlm import dlm_seasonal
from timemachines.evaluation import quick_brown_fox_randomized


def run_dlm():
    err = quick_brown_fox_randomized(f=dlm_seasonal, n=1000)


if __name__=='__main__':
    run_dlm()