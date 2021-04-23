import math


def weighted_average(y,w):
    return sum(wj * yj for yj, wj in zip(y, w)) / sum(w)


def normalize(w):
    return [ wj/sum(w) for wj in w]


def precision_weighted_skater(y,s,k,a,t,e,r=0.5):
    """ Not a skater in the usual sense since 'y' here takes a rather particular form,
        and it is intended to be used as with the ensemblefactory

       Treats y[1],y[3],y[5]... as unbiased estimates of y[0]
       Treats y[2],y[4],...     as std

       r -   determines the exponent:

             r=0.0 corresponds to equal weighting independent of x_std supplied by the models
             r=0.5 corresponds to simple weighting and is the default
             r->1  will use only the most accurate skater in the ensemble

    """
    tol = 1e-6  # minimum allowed x_std
    expon = 2*math.atanh(r)/math.atanh(0.5) if r<1-1e-6 else 10.0
    J  = int((len(y)-1)/2)
    y_stds = [ tol + y[2*j+2] for j in range(J) ]
    w  = normalize( [ 1./math.pow(y_std,expon) for y_std in y_stds ] )
    y_hats = [ y[2*j+1] for j in range(J) ]
    x = weighted_average(y=y_hats, w=w)
    x_std = min(y_stds)  # Conservative - they probably are not independent
    x_interp = [y[0]+(j+1)/k*(x-y[0]) for j in range(k)]
    x_std_interp = [math.sqrt(j+1)*x_std for j in range(k)]
    return x_interp, x_std_interp, {}
