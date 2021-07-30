from timemachines.skatertools.recommendations.suggestions import closest_fib


def test_closest():
    assert closest_fib(1.4)==1
    assert closest_fib(14) ==13
    assert closest_fib(-3) ==1
    assert closest_fib(1000) ==34


