from deap import benchmarks
import numpy as np
import math

# Some test objective functions to help guide optimizer choices
# -------------------------------------------------------------
#
# We'll use DEAP's set of groovy benchmarks, and landscapes package
# One day someone can explain to me why these benchmarks are defined on different domains when a cube would do.
# See pretty pictures at https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks
# Once can easily add more from DEAP or elsewhere
# For instance https://github.com/nathanrooy/landscapes#available-functions-from-single_objective


## Basis of tricky functions


def schwefel_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.schwefel
    u_squished = [ 1000*(ui**1.1-0.5) for ui in u ]
    try:
        return 0.001*benchmarks.schwefel(u_squished)[0]
    except Exception as e:
        raise Exception(e)


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




# By hand...

def rosenbrock_modified_on_cube(u:[float])->float:
    """ https://en.wikipedia.org/wiki/Rosenbrock_function """
    u_scaled = [4 * ui - 2 for ui in u]
    if len(u)==1:
        return (0.25-u_scaled[0])**2
    else:
        return np.sum([ 100*(ui_plus-ui*ui)+(1-ui)*(1-ui) for ui,ui_plus in zip(u_scaled[1:],u_scaled)])


# According to http://infinity77.net/global_optimization/test_functions.html#test-functions-index
# there are some really hard ones
# See https://github.com/andyfaff/ampgo/blob/master/%20ampgo%20--username%20andrea.gavana%40gmail.com/go_benchmark.py


def damavandi_on_cube(u:[float])->float:
    """ A trivial multi-dimensional extension of Damavandi's function """
    return damavandi2(u[0],u[1])


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


# Landscapes

from landscapes.single_objective import styblinski_tang, zakharov, salomon, rotated_hyper_ellipsoid, qing, michalewicz


def styblinski_tang_on_cube(u:[float])->float:
    u_scaled = [10*(ui-0.5) for ui in u]
    return styblinski_tang(u_scaled)


def zakharov_on_cube(u:[float])->float:
    u_scaled = [15*ui-10 for ui in u]
    return zakharov(u_scaled)


def salomon_on_cube(u:[float])->float:
    u_scaled = [200*ui - 100 for ui in u]
    return salomon(u_scaled)


def rotated_hyper_ellipsoid_on_cube(u:[float])->float:
    u_scaled = [2 * 65.536*ui - 65.536 for ui in u]
    return rotated_hyper_ellipsoid(u_scaled)


def qing_on_cube(u:[float])->float:
    u_scaled = [1000 * ui - 500 for ui in u]
    return qing(u_scaled)


def michaelewicz_on_cube(u:[float])->float:
    u_scaled = [4* ui - 2 for ui in u]
    return michalewicz(u_scaled,m=20)


LANDSCAPES_OBJECTIVES = [ styblinski_tang_on_cube, zakharov_on_cube, salomon_on_cube, rotated_hyper_ellipsoid_on_cube,
                          qing_on_cube, michaelewicz_on_cube ]


# Some copied from peabox
# https://github.com/stromatolith/peabox/blob/master/peabox/peabox_testfuncs.py
# as that isn't deployed to PyPI as far as I can determine


def ackley_on_cube(u:[float])->float:
    # allow parameter range -32.768<=x(i)<=32.768, global minimum at x=(0,0,...,0)
    rescaled_u = [2*32.768*ui - 32.768 for ui in u]
    x = np.asfarray(rescaled_u)
    ndim=len(x)
    a=20.; b=0.2; c=2.*math.pi
    return -a*np.exp(-b*np.sqrt(1./ndim*np.sum(x**2)))-np.exp(1./ndim*np.sum(np.cos(c*x)))+a+np.exp(1.)



AN_OBJECTIVE = deap_on_cube

DEAP_OBJECTIVES = [schwefel_on_cube, rastrigin_on_cube, griewank_on_cube, deap_on_cube ]
MISC_OBJECTIVES = [ paviani_on_cube, damavandi_on_cube, rosenbrock_modified_on_cube, ackley_on_cube ]

OBJECTIVES = DEAP_OBJECTIVES + LANDSCAPES_OBJECTIVES + MISC_OBJECTIVES


if __name__=="__main__":
    for objective in OBJECTIVES:
        objective(u=[0.5,0.5,0.5])
        objective(u=[0.5, 0.5, 0.0, 0.0, 0.0])