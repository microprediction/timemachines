from deap import benchmarks
import numpy as np
import math

# Some test objective functions to help guide optimizer choices
# -------------------------------------------------------------
#
# We'll use DEAP's set of groovy benchmarks, and landscapes package et al.
#
# See pretty pictures at https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks
# Some hardness assessment is at https://github.com/nathanrooy/landscapes#available-functions-from-single_objective but we'll do our own


## Basis of tricky functions

import datetime
DAY = datetime.datetime.today().day
OFFSET = DAY/50
POWER = 1+ (DAY % 3)/3.0
SHIFT = DAY/100


def smoosh(ui):
    """ Distort the interval to avoid obvious minima and avoid memorization """
    ui_rotate = ui+SHIFT % 1.0
    ui_shift = ui_rotate + SHIFT
    xi   = ui_shift**POWER
    low  = SHIFT**POWER
    high = (1+SHIFT)**POWER
    yi   = (xi-low)/(high-low)
    return yi**POWER


def schwefel_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.schwefel
    u_squished = [ 1000*(smoosh(ui)-0.5) for ui in u ]
    try:
        return 0.001*benchmarks.schwefel(u_squished)[0]/0.71063
    except Exception as e:
        raise Exception(e)


def griewank_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.griewank
    u_squished = [ 1200*(ui**1.1-0.5) for ui in u ]
    return benchmarks.griewank(u_squished)[0]/0.532075


def rastrigin_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.rastrigin
    u_squished = [ 10.24*(ui**1.1-0.5) for ui in u ]
    return 0.01*benchmarks.rastrigin(u_squished)[0]/0.059697


def bohachevsky_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.bohachevsky
    u_squished = [ 10*(ui**1.1-0.5) for ui in u ]
    return 1.0+benchmarks.bohachevsky(u_squished)[0]


def rosenbrock_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.rosenbrock
    u_squished = [ 200*(ui**1.1-0.5) for ui in u ]
    return 1+0.1*benchmarks.rosenbrock(u_squished)[0]/0.008949


def shaffer_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.schaffer
    u_squished = [ 200*(ui**1.1-0.5) for ui in u ]
    return 0.01*benchmarks.schaffer(u_squished)[0]/(0.1042133*0.71809)


def shekel_on_cube(u:[float])->float:
    # https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.schaffer

    n_dim = len(u)
    NUMMAX = 15
    A = 10 * np.random.rand(NUMMAX, n_dim)
    C = np.random.rand(NUMMAX)
    u_squished = [ 800*(smoosh(ui)-0.5) for ui in u ]
    return 1.2298-benchmarks.shekel(u_squished,A,C)[0]


## Combinations

def deap_combo1_on_cube(u:[float])->float:
    return 0.3*(schwefel_on_cube(u) + griewank_on_cube(u)+shekel_on_cube(u))/1.883


def deap_combo2_on_cube(u:[float])->float:
    return 0.5*(shaffer_on_cube(u) + shekel_on_cube(u))-0.1075


def deap_combo3_on_cube(u:[float])->float:
    return 0.5*(rosenbrock_on_cube(u) + bohachevsky_on_cube(u)+shekel_on_cube(u))/1.88


DEAP_OBJECTIVES = [schwefel_on_cube, rastrigin_on_cube, griewank_on_cube,
                   bohachevsky_on_cube, rosenbrock_on_cube, shaffer_on_cube, shekel_on_cube,
                   deap_combo1_on_cube, deap_combo2_on_cube, deap_combo3_on_cube]


# By hand...

def rosenbrock_modified_on_cube(u:[float])->float:
    """ https://en.wikipedia.org/wiki/Rosenbrock_function """
    u_scaled = [4 * ui - 2 for ui in u]
    if len(u)==1:
        return (0.25-u_scaled[0])**2
    else:
        return 5+0.001*np.sum([ 100*(ui_plus-ui*ui)+(1-ui)*(1-ui) for ui,ui_plus in zip(u_scaled[1:],u_scaled)])


# According to http://infinity77.net/global_optimization/test_functions.html#test-functions-index
# there are some really hard ones
# See https://github.com/andyfaff/ampgo/blob/master/%20ampgo%20--username%20andrea.gavana%40gmail.com/go_benchmark.py
# See also https://arxiv.org/pdf/1308.4008v1.pdf

def damavandi_on_cube(u:[float])->float:
    """ A trivial multi-dimensional extension of Damavandi's function """
    return 0.01*damavandi2(u[0],u[1])-0.46


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
    x = np.array([ 2.0001 + 5.996*smoosh(ui) for ui in u])
    return float( np.sum(np.log(x-2)**2.0 + np.log(10.0 - x)**2.0) - np.prod(x)**0.2 )/8.6456


# Landscapes

from landscapes.single_objective import styblinski_tang, zakharov, salomon, rotated_hyper_ellipsoid, qing, michalewicz


def styblinski_tang_on_cube(u:[float])->float:
    u_scaled = [10*(smoosh(ui)-0.5) for ui in u]
    return 3.3499+0.01*styblinski_tang(u_scaled)


def zakharov_on_cube(u:[float])->float:
    u_scaled = [15*smoosh(ui)-10 for ui in u]
    return 0.01*zakharov(u_scaled)/0.3462


def salomon_on_cube(u:[float])->float:
    u_scaled = [200*smoosh(ui) - 100 for ui in u]
    return salomon(u_scaled)/3.09999


def rotated_hyper_ellipsoid_on_cube(u:[float])->float:
    u_scaled = [2 * 65.536*smoosh(ui) - 65.536 for ui in u]
    return 0.1*rotated_hyper_ellipsoid(u_scaled)


def qing_on_cube(u:[float])->float:
    u_scaled = [1000*smoosh(ui) - 500 for ui in u]
    return qing(u_scaled)/0.01805


def michaelewicz_on_cube(u:[float])->float:
    u_scaled = [4*smoosh(ui) - 2 for ui in u]
    return 1.4439 + 0.1*michalewicz(u_scaled,m=20)


def landscapes_combo1_on_cube(u:[float])->float:
    return (qing_on_cube(u) + michaelewicz_on_cube(u))/(1.5744*1.4688)


def landscapes_combo2_on_cube(u:[float])->float:
    return (rotated_hyper_ellipsoid_on_cube(u) + salomon_on_cube(u))/(6.7555*0.82)


def landscapes_combo3_on_cube(u:[float])->float:
    return (2 + zakharov_on_cube(u) + styblinski_tang_on_cube(u))/4.4329


LANDSCAPES_OBJECTIVES = [ styblinski_tang_on_cube, zakharov_on_cube, salomon_on_cube, rotated_hyper_ellipsoid_on_cube,
                          qing_on_cube, michaelewicz_on_cube, landscapes_combo1_on_cube, landscapes_combo2_on_cube,
                          landscapes_combo3_on_cube ]

# Some copied from peabox
# https://github.com/stromatolith/peabox/blob/master/peabox/peabox_testfuncs.py
# as that isn't deployed to PyPI as far as I can determine


def ackley_on_cube(u:[float])->float:
    # allow parameter range -32.768<=x(i)<=32.768, global minimum at x=(0,0,...,0)
    rescaled_u = [2*32.768*smoosh(ui) - 32.768 for ui in u]
    x = np.asfarray(rescaled_u)
    ndim=len(x)
    a=20.; b=0.2; c=2.*math.pi
    return (-a*np.exp(-b*np.sqrt(1./ndim*np.sum(x**2)))-np.exp(1./ndim*np.sum(np.cos(c*x)))+a+np.exp(1.))/20.0



A_CLASSIC_OBJECTIVE = rastrigin_on_cube  # Just pick one for testing

MISC_OBJECTIVES = [ paviani_on_cube, damavandi_on_cube, rosenbrock_modified_on_cube, ackley_on_cube ]

CLASSIC_OBJECTIVES = DEAP_OBJECTIVES + LANDSCAPES_OBJECTIVES + MISC_OBJECTIVES


if __name__=="__main__":
    for objective in CLASSIC_OBJECTIVES:
        objective(u=[0.0,0.5,1.0])
        objective(u=[0.0, 0.5, 0.0, 0.0, 1.0])
    print(len(CLASSIC_OBJECTIVES))
