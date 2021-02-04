# timemachines [![Build Status](https://travis-ci.com/microprediction/timemachines.svg?branch=main)](https://travis-ci.com/microprediction/timemachines) ![tests](https://github.com/microprediction/timemachines/workflows/tests/badge.svg) ![regression](https://github.com/microprediction/timemachines/workflows/regression/badge.svg) ![offline-testing](https://github.com/microprediction/timemachines-testing/workflows/offline-testing/badge.svg)

A time series library, calling to other time series packages, where: 
  - Models are represented as mere *functions* 
  - Those functions suggest state machines for sequential consumption of observations (the state machines emit k-step ahead forecasts)
  - The collection of all hyper-parameters is squished into a single point, in (0,1).  

Some functionality is drawn from:
  - fbprophet, 
  - pydlm, 
  - pmdarima,
  - more from the [listing of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) and, 
  - some home-grown approaches too. 

We also [test](https://github.com/microprediction/timemachines-testing) models and optimizer combinations. To that end, 
this package exposes some (but not all) functionality from numerous global optimizers in a consistent manner. Perhaps that 
is of independent interest. It is easy to exploit:
 
  - scipy
  - ax-platform
  - hyperopt
  - optuna
  - platypus
  - pymoo
  - pysot
  - shgo
  - swarmlib
  - and possible others. See [global optimizers](https://github.com/microprediction/timemachines/tree/main/timemachines/optimizers) for the full list. 
 
A third distinguishing feature of this library is that it is trained on [live data](https://www.microprediction.org/browse_streams.html). This
is constantly updating, so the temptation to overfit to a stale, fixed history is reduced.  

# Skaters

A time series approach manifests as a "skater", or we should say a function with a "skater" signature, that is
considered a recipe for a *state machine*. The function is intended to be applied repeatedly. For example one could harvest
a sequence of the model predictions as follows:

    def posteriors(f,y):
        s = {}       
        x = list()
        for yi in y: 
            xi, xi_std, s = f(yi,s)
            x.append(xi)
        return x
 
Notice that f here is just a *single function*. There are no classes in this package - well except for the ones used as hacks to suppress optimizers who yabber incesantly. The callee f will
create state 's' if it infers that this is the first time it is being called (note the empty dict passed). So
long as the caller sends the callee an empty dict the first time, and 's' on subsequent invocations as shown above, everything should go swimmingly.  
 
### Observations: exogenous versus known-in-advance
 
- The skater function *f* takes a vector *y*, where the quantity to be predicted is y[0] and there may be other, simultaneously observed
 variables y[1:] whose lags may be helpful in predicting y[0].  
- The function also takes a quantity *a* which is a vector of numbers known k-steps in advance. 

      x, w, s = f(   y:Union[float,[float]],               # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:[float]=None,                          # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for hype(r)-pa(r)amete(r)s in R^n)

### If you read nothing else here...
This is important: 

- The caller should provide *a* pertaining to k-steps ahead, not the contemporaneous 'a'.  
- The caller should provide *a* pertaining to k-steps ahead, not the contemporaneous 'a'.  
- The caller should provide *a* pertaining to k-steps ahead, not the contemporaneous 'a'.  

### Expiry

The use of *e* is a fairly *weak convention* that many skaters ignore. In theory, a large expiry *e* can be used as a hint to the callee that
 there is time enough to do a 'fit', which we might define as anything taking longer than the usual function invocation.
 However, this is between the caller and it's priest really - or its prophet. Some skaters, such
 as the prophet skater, do a full 'fit' every invocation so this is meaningless. Other skaters
  no separate notion of 'fit' versus 'update' because everything is incremental. 
   

### Return values

Morally, a skater *suggests* an affine transformation of the incoming data. For each prediction horizon it returns
two numbers where the first can be *interpreted* as a point estimate (but need not be) and the second is *typically* suggestive
of a symmetric error std, or width. 


          -> x     [float],    # A vector of point estimates, or anchor points, or theos
             x_std [float]     # A vector of "scale" quantities (such as a standard deviation of expected forecast errors) 
             s    Any,         # Posterior state, intended for safe keeping by the callee until the next invocation 
                       

In returning state, one possible intent is that the *caller* might carry the state from one invocation to the next, not the *callee*. This
is arguably more convenient than having the predicting object maintain state, because the caller can "freeze" the state as they see fit, as 
when making conditional predictions. It also eyes lambda-based deployments and *encourages* tidy use of internal state - not that we succeed
 when calling down to statsmodels (but all the home grown models here use simple dictionaries, making serialization trivial).

### Skater hyper-parameters
 
We use a further, somewhat unusual convention. All model hyper-parameters, should they exist, are squished down into
 a *scalar* quantity *r*. This imposes at skater "design time" a consistent hyper-parameter space. This step may seem
  unnatural, but it facilitates comparisons of models and hyper-parameter optimizers in different settings. 
  It is workable, we hope, with some space-filling curve conventions. More on that below. 
  
 
### Obligatory picture of a skater


![](https://i.imgur.com/DkZvZRq.png)

Photography by [Joe Cook](https://www.instagram.com/joecooke_/?utm_medium=referral&utm_source=unsplash)

    
## Summary of conventions: 

- State
    - The caller, not the callee, persists state from one invocation to the next
    - The caller passes s={}} the first time, and the callee initializes state
    - State can be mutable for efficiency (e.g. it might be a long buffer) or not. 
    - State should, ideally, be JSON-friendly. 
       
- Observations: target, and contemporaneous exogenous
     - If y is a vector, the target is the first element y[0]
     - The elements y[1:] are contemporaneous exogenous variables, *not known in advance*.  
     - Missing data can be supplied to some skaters, as np.nan.  
     - Most skaters will accept scalar *y* and *a* if there is only one of either. 
    
- Variables known k-steps in advance, or conditioning variables:
     - Pass the *vector* argument *a* that will occur in k-steps time (not the contemporaneous one)
     - Remark: In the case of k=1 there are different interpretations that are possible beyond "business day", such as "size of a trade" or "joystick up" etc. 

- Hyper-Parameter space:
     - A float *r* in (0,1). 
     - This package provides functions *to_space* and *from_space*, for expanding to R^n using space filling curves, so that the callee's (hyper) parameter optimization can still exploit geometry, if it wants to.   
     - See discussion below    
    
    
Nothing here is put forward
   as *the right way* to write time series packages - more a way of exposing their functionality for comparisons. 
  If you are interested in design thoughts for time series maybe participate in this [thread](https://github.com/MaxBenChrist/awesome_time_series_in_python/issues/1). 

### Conventions for hyper-params in the unit interval or hypercube

The restriction that all hyper-parameters be represented as r in (0,1) seems harsh. To be slightly less harsh, we include some standard ways
to use (0,1)^2 or (0,1)^3 should that be preferable. Admittedly, this may still not be the most natural way to represent choices, but here
we are trying to give lots of different optimizers a run at the problem. Of course, if you want to optimize skaters by some other means
and want a flexible way to represents searches then use one of the optim packages directly (e.g. optuna is pretty flexible in this regard). On 
the other hand, if you want to benefit from the operational simplicity of r in (0,1) then...

- The default map *from_space* from (0,1)^3 or (0,1)^2->(0,1) uses interleaving of digits in the binary representations (after first scaling).
- The script [demo_param_ordering.py](https://github.com/microprediction/timemachines/blob/master/examples/demo_param_ordering.py) illustrates
the mapping from r in (0,1) to R^n demonstrating that the first coordinate will vary
more smoothly as we vary r than the second, and so on.  
- If you need dim>3 for hyper-parameters, you can always use *to_space* or *from_space* more than once. 
- There are some functions provided to help you squish your hyper-params into the hypercube. The script [demo_balanced_log_scale.py](https://github.com/microprediction/timemachines/blob/master/examples/demo_balanced_log_scale.py) illustrates a
quasi-logarithmic parameter mapping from r in (0,1) to R which you can take or leave. 

[![IMAGE ALT TEXT](https://i.imgur.com/4F1oHXR.png)](https://vimeo.com/497113737 "Parameter importance")
Click to see video
 
    
### FAQ 1: Why not have the model persist the state?

Answer: Well, you can trivially turn any skater function into a callable that does that, should you wish: 

       class Predictor:
   
           def __init__(self,f):
                self.f = f
                self.s = s

           def __call__(self,y,k,a,t,e):
                x, x_std, self.s = self.f(y=y,s=self.s,k=k,a=a,t=t,e=e)
                return x, x_std

or write a decorator. Whatever, it's Python. 

Answer: The intent is to produce simple lambda-friendly models and,

- a *reasonable* way to map the most important hyper-parameter choices (we hope),
- that imposes some geometric discipline on the hyper-parameter space in the first place, and
- facilitates comparison of different ways to search hyper-parameters, across packages which have *entirely different conventions* and hyper-parameter spaces. 

### FAQ 2: Why not use the packages, like prophet, directly?

Answer: Maybe you should. Observe that this package wraps *some* functionality, not all by any means. You should use the original
packages for maximum flexibility. However, as noted, you *might* like this package if you want to be able to do this:

        s,k = {}, 3
        for yi,ai in zip(y,a[k:]): 
            xi, xi_std, s = f(y=yi,s=s,k=k,a=ai)

Notice what isn't here: 
 - Pandas dataframes
 - A long list of methods and properties 
 - Column naming conventions 
 - The customary 10-50 lines of setup code before a prediction can be made,
 - The customary need to trace into the code to infer intent, including which parameters are supposed to be supplied. 
 - Possible confusion between variables known in advance and those observed contemporaneously,
 - Possible confusion about prediction horizon,
 - Possible conflation of 3-step ahead prediction with the 1-step ahead prediction applied three times, 
 - Datetime manipulation, and conventions like '5min' which not everyone agrees on. 

There are also limitations of the skater approach. The simple data model in *y*, *a* is not well suited to problems where exogenous data comes and goes, and therefore cannot
easily be represented by a vector of fixed length (you might consider a dictionary interface instead, as with
the river package). 

### FAQ 3: Only Point Estimates and Confidence Intervals?  

Yes, the skater does not return a full distribution - unless you smuggle it into the state. 
However this package was motivated by the desire to create better free turnkey distributional forecasts, at [microprediction.org](https://www.microprediction.org), and you might infer that skaters returning two numbers per horizon might be useful 
as part of a chain of computations that eventually produces a distributional estimate. Skaters can be considered linear transforms
of incoming data, and part of the agenda here is figuring out how to judge skaters
in a manner that better reflects downstream use in distributional estimates. Here the theory of proper scoring rules doesn't really suffice, it would seem. End of aside.  
 

### Contributing 
If you'd like to contribute to this standardizing and benchmarking effort, here are some ideas:

- See the [list of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) ranked by download popularity. 
- Think about the most important hyper-parameters and consider "warming up" the mapping (0,1)->hyper-params by testing on real data. There is a [tutorial](https://www.microprediction.com/python-3) on retrieving live data, or use the [real data](https://pypi.org/project/realdata/) package, if that's simpler.
- The [comparison of hyper-parameter optimization packages](https://www.microprediction.com/blog/optimize) might also be helpful.  

If you are the maintainer of a time series package, we'd love your feedback and if you take the time to submit a PR here that incorporates your library, do yourself a favor and also enable "supporting" on your repo. 

