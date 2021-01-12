from deap import benchmarks
import math
import numpy as np

# Some test objective functions to help guide optimizer choices
# -------------------------------------------------------------
#
# We'll use DEAP's set of groovy benchmarks.
# Your mileage may vary
# See pretty pictures at https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks
# Once can easily add more from DEAP or elsewhere

## Basis of tricky functions


def schwefel_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.schwefel
    u_squished = [ 1000*(ui**1.1-0.5) for ui in u ]
    return 0.001*benchmarks.schwefel(u_squished)[0]


def griewank_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.griewank
    u_squished = [ 1200*(ui**1.1-0.5) for ui in u ]
    return benchmarks.griewank(u_squished)[0]


def rastrigin_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.rastrigin
    u_squished = [ 10.24*(ui**1.1-0.5) for ui in u ]
    return 0.01*benchmarks.rastrigin(u_squished)[0]

## Combinations

def deap_on_cube(u:[float])->float:
    return schwefel_on_cube(u) + griewank_on_cube(u) + rastrigin_on_cube(u)


# According to http://infinity77.net/global_optimization/test_functions.html#test-functions-index
# there are some really hard ones
# See https://github.com/andyfaff/ampgo/blob/master/%20ampgo%20--username%20andrea.gavana%40gmail.com/go_benchmark.py


def damavandi_on_cube(u:[float])->float:
    """ A multi-dimensional extension of Damavandi's function """
    u0 = u[0]
    dama = [ damavandi2(u0,u_) for u_ in u[1:] ]
    return float(np.sum(dama))


def damavandi2(u1,u2)->float:
    """ Pretty evil function this one """
    # http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Damavandi
    x1 = u1/14.
    x2 = u2/14.
    numerator = math.sin(math.pi*(x1 - 2.0))*math.sin(math.pi*(x2 - 2.0))
    denumerator = (math.pi**2)*(x1 - 2.0)*(x2 - 2.0)
    factor1 = 1.0 - (abs(numerator / denumerator))**5.0
    factor2 = 2 + (x1 - 7.0)**2.0 + 2*(x2 - 7.0)**2.0
    return factor1*factor2


def paviani_on_cube(u:[float])->float:
    # http://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Paviani
    x = np.array([ 2.001+(ui**1.1)*6.996 for ui in u])
    return float( np.sum(np.log(x-2)**2.0 + np.log(10.0 - x)**2.0) - np.prod(x)**0.2 )


AN_OBJECTIVE = deap_on_cube
OBJECTIVES   = [schwefel_on_cube, rastrigin_on_cube, griewank_on_cube, deap_on_cube,
                paviani_on_cube, damavandi_on_cube ]


if __name__=="__main__":
    for objective in OBJECTIVES:
        objective(u=[0.5,0.5,0.5])