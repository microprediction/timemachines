# timemachines ![tests](https://github.com/microprediction/timemachines/workflows/tests/badge.svg)![tsa](https://github.com/microprediction/timemachines/workflows/test-tsa/badge.svg) ![sktime](https://github.com/microprediction/timemachines/workflows/test-sktime/badge.svg) ![tbats](https://github.com/microprediction/timemachines/workflows/test-tbats/badge.svg) ![simdkalman](https://github.com/microprediction/timemachines/workflows/test-simdkalman/badge.svg) ![prophet](https://github.com/microprediction/timemachines/workflows/test-prophet/badge.svg) ![orbit](https://github.com/microprediction/timemachines/workflows/test-orbit/badge.svg)  ![neuralprophet](https://github.com/microprediction/timemachines/workflows/test-neuralprophet/badge.svg) ![pmd](https://github.com/microprediction/timemachines/workflows/test-pmd/badge.svg) ![pydlm](https://github.com/microprediction/timemachines/workflows/test-pydlm/badge.svg) ![river](https://github.com/microprediction/timemachines/workflows/test-river/badge.svg) ![divinity](https://github.com/microprediction/timemachines/workflows/test-divinity/badge.svg)![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Fast, incremental, time-series prediction
... and some slow ones. Use popular forecasting packages with one line of code. Or just browse their [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) to help decide which to try out first. There's also a recommendation [colab notebook](https://github.com/microprediction/timeseries-elo-ratings/blob/main/time_series_recommendations.ipynb) you can open and run. 

What's different:

   - **Simple k-step ahead forecasts in functional style** There are no "models" here requiring setup, only forecast functions:
       
          x, x_hat, s = f(y,s,k)
       
       These functions are called skaters. 

   - **Simple canonical use** of *some* functionality from packages like [river](https://github.com/online-ml/river), [pydlm](https://github.com/wwrechard/pydlm), [tbats](https://github.com/intive-DataScience/tbats), [pmdarima](http://alkaline-ml.com/pmdarima/), [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html), [neuralprophet](https://neuralprophet.com/), Facebook [Prophet](https://facebook.github.io/prophet/), 
   Uber [orbit](https://eng.uber.com/orbit/) and more. 

   - **Simple fast accurate alternatives** to popular time series packages. For example the [thinking](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/thinking.py) skaters perform well in the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html), and much better than the brand names. See the  [article](https://www.microprediction.com/blog/timemachines) comparing them to Facebook prophet and Neural Prophet.

   - **Ongoing, incremental, empirical evaluation**. Again, see the [leaderboards](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) produced by
    the accompanying repository [timeseries-elo-ratings](https://github.com/microprediction/timeseries-elo-ratings). Assessment is always out of sample and uses *live*, constantly updating real-world data 
     from [microprediction.org](https://www.microprediction.org/browse_streams.html).   
    
   - **Simple stacking, ensembling and combining** of models. The function form makes it easy to do this
   with one line of code, quite often (again, see [thinking.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/thinking.py) for an illustration, 
   or [prophetskaterscomposed.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/proph/prophskaterscomposed.py)).

  - **Simpler deployment**. There is no state, other that that explicitly returned to the caller. For skaters relying only on the timemachines and river packages (the fast ones), the state is a pure Python dictionary trivially converted to JSON and back (for instance in a web application). See the [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) for a little more discussion.   

**NO CLASSES**  **NO DATAFRAMES** **NO CEREMONY**   

Nothing to slow you down!

To emphasize, in this package a time series "model" is a plain old function taking scalars and lists as arguments. Those functions have a "skater" signature, facilitating "[skating](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py)".
   One might say that skater functions *suggest* state machines for sequential assimilation of observations (as a data point arrives, 
    forecasts for 1,2,...,k steps ahead, with corresponding standard deviations are emitted). However here the *caller* is expected to maintain state from one 
    invocation (data point) to the next. See the [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) if this seems odd. 

### New contributor guide:
    
See  [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md)

## Install

The suggested install is:  

    pip install --upgrade pip
    pip install --upgrade numpy
    pip install timemachines

Then check the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) to decide which
packages you want to utilize - they aren't in by default. 

    pip install --upgrade river 
    pip install --upgrade simdkalman
    pip install --ugprade sktime
    pip install --upgrade tbats
    pip install --upgrade orbit-ml
    pip install --upgrade pydlm
    pip install --upgrade divinity
    pip install --upgrade pmdarima
    pip install --upgrade prophet
    pip install --upgrade neuralprophet
    
Then add matplotlib if you want to use plotting utilities provides

    pip install matplotlib 

And add microprediction if you want to use live data

    pip install microprediction   
    
I'm reluctant to put anything beyond statsmodels in the timemachines package requirements until there
is statistical justification. See my [review of prophet](https://www.microprediction.com/blog/prophet) for example, which
is seemingly both slow and innacurate (but on the other hand, river won't slow you down). By the way, the apple m1 install situation is fluid. I'd suggest you first get numpy, cython, pandas to work. 
You might try adding the pip argument to skip pep517 if you run into trouble:
 
    pip install whatever --no-use-pep517


### Quick start 

My hope is that the [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py) utilities also
serve as demonstrations of how to use any given skater in this library. If f is a skater then you call it repeatedly:

    from timemachines.skaters.simple.thinking import thinking_slow_and_fast 
    import numpy as np
    y = np.cumsum(np.random.randn(1000))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = thinking_slow_and_fast(y=yi, s=s, k=3)
        x.append(xi)
     
This will accumulate 3-step ahead prediction vectors. Or to plot actual data:

    from timemachines.skaters.simple.thinking import thinking_slow_and_slow
    from timemachines.skatertools.visualization.priorplot import prior_plot
    from timemachines.skatertools.data.real import hospital
    import matplotlib.pyplot as plt
    y = hospital(n=200)
    prior_plot(f=thinking_slow_and_slow,y=y)
    plt.show()
  
There's more in [examples/basic_usage](https://github.com/microprediction/timemachines/tree/main/examples/basic_usage).
  
![](https://i.imgur.com/elu5muO.png)
  
## The Skater signature 

Okay, here's a little more about "skater" functions. I'm repeating myself somewhat but the good thing is, this is the only thing you need to know. Morally this package is a mere collection of skater functions and they all operate like this: 

      x, w, s = f(   y:Union[float,[float]],             # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:[float]=None,                          # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for hype(r)-pa(r)amete(r)s in R^n)

Evidently, the function is intended to be applied repeatedly. For example one could harvest
a sequence of the model predictions as follows:

    def posteriors(f,y):
        s = {}       
        x = list()
        for yi in y: 
            xi, xi_std, s = f(yi,s)
            x.append(xi)
        return x
 
or see the prominently positioned [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py). Notice the use of s={} on first invocation. 
 
### Skater "y" argument

A skater function *f* takes a vector *y*, where the quantity to be predicted is y[0] and there may be other, simultaneously observed
 variables y[1:] deemed helpful in predicting y[0].

### Skater "s" argument
 
The state. The convention is that the caller passes the skater an empty dict on the first invocation, or to reset it. Thus the callee must initialize state if it receives an empty dictionary. It should return to the caller anything it will need for the next invocation. Skaters are pure in that sense.  

### Skater "k" argument 

Determines the length of the term structure of predictions (and also their standard deviations) that will be returned. This cannot be varied from
one invocation to the next. 

### Skater "a" argument 

A vector of known-in-advance variables. You can also use the "a" argument for conditional prediction. This is a nice advantage of keeping skaters pure - though the caller might need to make a copy of the prior state if she intends to reuse it. 

### Skater "t" argument 

Epoch time of the observation. 

### Skater "e" argument ("expiry")

Suggests a number of seconds allowed for computation, though skater's don't necessarily comply. See remarks below. 
   
## Skater "r" argument ("hype(r) pa(r)amete(r)s")

A scalar in the closed interval \[0,1\] represents *all* hyper-parameters. See comments below. 

### Return values

Two vectors and the posterior state. The first set of *k* numbers can be *interpreted* as a point estimate (but need not be) and the second is *typically* suggestive of a symmetric error std, or width. However a broader interpretation is possible wherein a skater *suggests* a useful affine transformation of the incoming data and nothing more.  


          -> x     [float],    # A vector of point estimates, or anchor points, or theos
             x_std [float]     # A vector of "scale" quantities (such as a standard deviation of expected forecast errors) 
             s    Any,         # Posterior state, intended for safe keeping by the callee until the next invocation 
                       

In returning state, the intent is that the *caller* might carry the state from one invocation to the next verbatim. This is arguably more convenient than having the predicting object maintain state, because the caller can "freeze" the state as they see fit, as 
when making conditional predictions. This also eyes lambda-based deployments and *encourages* tidy use of internal state - not that we succeed
 when calling down to statsmodels (though prophet, and others including the home grown models use simple dictionaries, making serialization trivial). 
 
You'll notice also that parameter use seems limited. This is deliberate. A skater is morally a "bound" model (i.e. fixed hyper-parameters) and ready to use. Any fitting, estimation or updating is the skater's internal responsibility. That said, it is sometimes useful to enlarge the skater concept to include hyper-parameters, as this enourages a more standardized way to expose and fit them. It remains the responsibility of the skater designer to ensure that the parameter space is folded into (0,1) is a somewhat sensible way. 

The use of a single scalar for hyper-parameters may seem unnatural, but is slighly less unnatural if [conventions](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/utilities/conventions.py) are followed that inflate \[0,1\] into the square \[0,1\]^2 or the cube \[0,1\]^3. See the functions **to_space** and **from_space**. This also makes it trivial for anyone to design black box optimization routines that can work on any skater, without knowing its working. The humpday package makes this trivial - albeit time-consuming. 

### More on the e argument ...

The use of *e* is a fairly weak convention. In theory:

   - A large expiry *e* can be used as a hint to the callee that
 there is time enough to do a 'fit', which we might define as anything taking longer than the usual function invocation. 
   - A negative *e* suggests that there isn't even time for a "proper" prediction to be made, never mind a model fit. It suggests that we are still in a burn-in period where the caller doesn't care too much, if at all, about the quality of prediction. The callee (i.e. the skater) should, however, process this observation *somehow* because this is the only way it can receive history. There won't be another chance. Thus some skaters will use e<0 as a hint to dump the obervation into a buffer so it can be used in the next model fit. They return a naive forecast, confident that this won't matter.  
 
Some skaters are so fast that a separate notion of 'fit' versus 'update' is irrelevant. Other skaters will periodically fit whether or not e>0 is passed. 

The "e" conventions are useful for testing and assessment. You'll notice that the Elo rating code passes a sequence of e's something looking like:

     -1, -1, -1, ... -1 1000 1000 1000 1000 1000 ...
     
because it wants to allow the skaters to receive some history before they are evaluated. On the other hand, waiting for Facebook prophet to fit itself 500 times is a bit like waiting for the second coming of Christ. 
    
### Summary of conventions: 

- State
    - The caller, not the callee, persists state from one invocation to the next
    - The caller passes s={} the first time, and the callee initializes state
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
     
See [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) or file an issue if anything offends you greatly. 
 
### Related illustrations 

- See [examples](https://github.com/microprediction/timemachines/tree/main/examples) 

### Tuning hyper-params

It's also dead easy (though possibly time-consuming) to hyper-optimize skaters offline. By convention they only admit a single hyper-parameter, if any. This means you can, with [one line of code](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/tuning/hyper.py) use [HumpDay](https://github.com/microprediction/humpday) to call down to scipy.optimize, ax-platform,
   hyperopt, optuna, platypus, pymoo, pySOT, skopt, dlib, nlopt, bayesian-optimization, nevergrad or your favourite black-box optimizer.  

- See [examples/tuning](https://github.com/microprediction/timemachines/tree/main/examples/tuning)
- See [tuning](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools/tuning)
    
## Contribute 

If you'd like to contribute to this standardizing and benchmarking effort, here are some ideas:

- Read the  [contributor guide](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md)
- See the [list of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) ranked by download popularity. 
- Think about the most important hyper-parameters and consider "warming up" the mapping (0,1)->hyper-params by testing on real data. There is a [tutorial](https://www.microprediction.com/python-3) on retrieving live data, or use the [real data](https://pypi.org/project/realdata/) package, if that's simpler.
- The [comparison of hyper-parameter optimization packages](https://www.microprediction.com/blog/optimize) might also be helpful.  
- Read the  [contributor guide](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md)

If you are the maintainer of a time series package, we'd love your feedback and if you take the time to submit a PR here
 that incorporates your library, do yourself a favor and also enable "supporting" on your repo. Nothing here is put forward
   as *the right way* to write time series packages - more a way of exposing their functionality for comparisons. 
  If you are interested in design thoughts for time series maybe participate in this [thread](https://github.com/MaxBenChrist/awesome_time_series_in_python/issues/1). 

## FAQ 
See [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md)
