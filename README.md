# timemachines [![Build Status](https://travis-ci.com/microprediction/timemachines.svg?branch=main)](https://travis-ci.com/microprediction/timemachines) ![tests](https://github.com/microprediction/timemachines/workflows/tests/badge.svg) ![regression](https://github.com/microprediction/timemachines/workflows/regression/badge.svg)



This package is an experiment in a different approach to the representation of time series models. Here a time series model:

- takes the form of a *pure function* with a *skater* signature,
- that is a recipe for a *state machine*,
- where the intent that the *caller* might carry the state from one invocation to the next, not the *callee*, and
- with the further, somewhat unusual convention that variables known in advance (*a*) and the full set of model hyper-parameters (*r*) are both squished down into their respective *scalar* arguments. 

The penultimate convention is for generality, and also eyes lambda-based deployments. The last convention imposes at design time a consistent hyper-parameter space. This step may seem unnatural, but it facilitates comparisons of models and hyper-parameter optimizers in different settings. It is workable, we hope, with some space-filling curve conventions.   

### Want to discuss time series modeling standardization?

This isn't put forward as *the right way* to write time series packages - more a way of exposing their functionality. If you are interested in design thoughts for time series maybe participate in this thread https://github.com/MaxBenChrist/awesome_time_series_in_python/issues/1. 

### A "skater" function 

Most time series packages use a complex combination of methods and data to represent a time series model, its fitting, and forecasting usage. But in this package a "model" is *merely a function* We mean *function* in the mathematical sense.   

    x, s, w = f(   y:Union[float,[float]],               # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:float=None,                            # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for hype(r)-pa(r)amete(r)s in R^n)
The function returns: 

                     -> float,                           # A point estimate, or anchor point, or theo
                        Any,                             # Posterior state, intended for safe keeping by the callee until the next invocation 
                        Any                              # Everything else (e.g. confidence intervals) not needed for the next invocation. 
                
(Yes one might quibble with the purity given that state s can be modified, but that's Python sensible).  

### Skating forward

    def posteriors(f,ys):
        s = None
        xs = list()
        for y in ys: 
            x, s, _ = f(y,s)
            xs.append(xs)
        return xs

![](https://i.imgur.com/DkZvZRq.png)

Picture by [Joe Cook](https://www.instagram.com/joecooke_/?utm_medium=referral&utm_source=unsplash)

    
### Conventions: 

- The caller, not the callee, persists state from one invocation to the next
    - The format taken by state is determined by the callee, not caller
    - The caller passes s=None the first time
    - The function initializes state as necessary, and passes it back
    - The caller keeps the state and sends it back to the callee
    - State can be mutable for efficiency (e.g. it might be a long buffer) or not. 
    - Recall that Python is pass-by-object-reference. 
    - State should, ideally, be JSON-friendly. Use .tolist() on arrays.
    - State is not an invitation to sneak in additional arguments.
       
- Univariate or multivariate observation argument
     - If y is a vector, the target is the first element y[0]
     - The elements y[1:] are contemporaneous exogenous variables, *not known in advance*.  
     - Missing data as np.nan but *not None* (see fitting below)

- Fitting:  
     - If y=None is passed, it is a suggestion to the callee to perform fitting, should that be necessary. 
     - Or some other offline periodic task. 
     - In this case the *e* argument takes on a slightly different interpretation, and should probably
     be considerably larger than usual. 
     - The callee should return x=None, as acknowledgement that it has recognized the "offline" convention
   
- Variables known in advance, or conditioning variables:
     - Passed as *scalar* argument *a* in (0,1). 
     - See discussion below re: space-filling curves so you know this isn't really a huge restriction.  
     - Rationale: make it easier to design general purpose conditional prediction algorithms
     - Bear in mind many functions will ignore this argument, so we have little to lose here. 
     - Caller can deepcopy the state to effect multiple conditional predictions.
     - Example: business day indicator
     - Example: size of a trade
     - Example: joystick button up 

- Parameter space:
     - Caller has a limited ability to suggest variation in parameters (or maybe hyper-parameters, since 
     many callees will fit parameters on the fly or when there is time).
     - This communication is squished into a single float *r* in (0,1). 
     - Arguably, this makes callees more canonical and, 
     - seriously, there are lots of real numbers, and 
     - the intent here is that the caller shouldn't need to know a lot about parameters.
     - This package provides some conventions for expanding to R^n using space filling curves,
     - so that the callee's (hyper) parameter optimization can still exploit geometry, as you see fit. 
      
- Ordering of parameters in space-filling curve:
    - The most important variables should be listed first, as they vary more slowly. 
    - See picture below or video
    
### Space-filling conventions for *a* and *r*

The script [demo_balanced_log_scale.py](https://github.com/microprediction/timemachines/blob/master/examples/demo_balanced_log_scale.py) illustrates the
quasi-logarithmic parameter mapping from r in (0,1) to R. 

The script [demo_param_ordering.py](https://github.com/microprediction/timemachines/blob/master/examples/demo_param_ordering.py) illustrates
the mapping from r in (0,1) to R^n. Observe why the most important parameter should be listed first. It will vary
more smoothly as we vary r. 

[![IMAGE ALT TEXT](https://i.imgur.com/4F1oHXR.png)](https://vimeo.com/497113737 "Parameter importance")
Click to see video
 
    
### FAQ:

Question 1. Why not have the model persist the state?

Answer 1. Go ahead:

       class Predictor:
   
           def __init__(self,f):
                self.f = f
                self.s = s

           def __call__(self,y,k,a,t,e):
                x, self.s = self.f(y=y,s=self.s,k=k,a=a,t=t,e=e)
                return x

or write a decorator. However:
- We have lambda patterns in mind
- The callee has more control in this setup (e.g. for multiple conditional forecasts)

Question 2. Why do it this bare-bones manner with squished parameter spaces?  

Answer 2. The intent is to produce lambda-friendly models but also:
- Comparison, combination and search for models, made possible by
- A *reasonable* way to map the most important hyper-parameter choices (we hope),
- Which imposes some geometric discipline on the hyper-parameter space (e.g. most important first), and
- enables search across packages which have *entirely different conventions* and hyper-parameter spaces. 


Observe that this package wraps *some* partial functionality of some time series prediction libraries. Those libraries could not be further removed from the above in that they:
 - Use pandas dataframes
 - Bundle data with prediction logic
 - Rely on column naming conventions 
 - Require 10-20 lines of setup code before a prediction can be made
 - Require tracing into the code to infer intent
 - Use conventions such as '5min' which not everyone agrees on 

This package should *not* be viewed as an attempt to wrap most of the functionality of these packages. If you 
have patterns in mind that match them, and you are confident of their performance, you are best served to 
use them directly. 

### Scope and limitations
The simple interface is not well suited to problems where exogenous data comes and goes. 
You might consider a dictionary interface instead, as with the river package. 
It is also not well suited to fixed horizon forecasting if the data isn't sampled terribly regularly. 
Nor is it well suited to prediction of multiple time series whose sampling occurs irregularly. 
Ordinal values can be kludged into the parameter space and action argument, but purely categorical not so much. And finally, if you
don't like the idea of hyper-parameters lying in R^n or don't see any obvious embedding, this might 
not be for you. 

### Yes, we're keen to receive PR's
If you'd like to contribute to this standardizing and benchmarking effort, here are some ideas:

- See the [list of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) ranked by download popularity. 
- Think about the most important hyper-parameters.
- Consider "warming up" the mapping (0,1)->hyper-params by testing on real data. There is a [tutorial](https://www.microprediction.com/python-3) on retrieving live data, or use the [real data](https://pypi.org/project/realdata/) package, if that's simpler.
- The [comparison of hyper-parameter optimization packages](https://www.microprediction.com/blog/optimize) might also be helpful.  

If you are the maintainer of a time series package, we'd love your feedback and if you take the time to submit a PR here, do yourself a favor and also enable "supporting" on your repo. 

### Deployment

Some of these models are used as intermediate steps in the creation of distributional forecasts, at [microprediction.org](https://www.microprediction.org). 
