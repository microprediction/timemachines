from timemachines.skaters.divine import divinity_seasonal
from timemachines.evaluation import quick_brown_fox


def test_divine():
    err = quick_brown_fox(f=divinity_seasonal, n=55) # Won't get to burn-in

