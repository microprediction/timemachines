from timemachines.skaters.divine import divine
from timemachines.evaluation import quick_brown_fox


def test_divine():
    err = quick_brown_fox(f=divine,n=75)

