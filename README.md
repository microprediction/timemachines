# timemachines ![tests](https://github.com/microprediction/timemachines/workflows/tests/badge.svg) ![skater-elo-ratings](https://github.com/microprediction/timemachines-testing/workflows/skater-elo-ratings/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

### Popular time-series packages in a simple functional form

What's different:
  - Simple functional form. Time series "models" are literally functions, with a "skater" signature explained below. These functions suggest state machines that will perform sequential consumption of observations. The state machines emit vectors of forecasts of length *k*, and also standard deviations. NO CLASSES. NO DATAFRAMES. NO CEREMONY. 
  - Simple combinations. Composition and combining in other ways is trivial. 
  - Simple tuning using virtually any optimization library. Skaters with *r* parameter are yet to be "fixed". They use a standardized hyper-parameter space (the cube) so that, using the the [HumpDay](https://github.com/microprediction/humpday) package, one can easily switch from from scipy.optimize to ax-platform or hyperopt, optuna, platypus, pymoo, pySOT, skopt, bayesian-optimization, nevergrad and more ... including close to 100 global optimization strategies. 
  - Elo ratings of time series model instances (those with fixed hyper-params) populate [leaderboards](https://github.com/microprediction/timemachines-testing/tree/main/skater_elo_ratings/leaderboards) in the accompanying repository [timemachines-testing](https://github.com/microprediction/timemachines-testing). Those Elo ratings based on head to head battles on live, constantly refreshing data like at [microprediction.org](https://www.microprediction.org/browse_streams.html) - thereby discouraging hyperparameter overfiting.  

![](https://i.imgur.com/elu5muO.png)

A skater function *f* takes a vector *y*, where the quantity to be predicted is y[0] and there may be other, simultaneously observed
 variables y[1:] deemed helpful in predicting y[0]. The function also takes a quantity *a* which is a vector of numbers known k-steps in advance. 

      x, w, s = f(   y:Union[float,[float]],               # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:[float]=None,                          # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for hype(r)-pa(r)amete(r)s in R^n)

The function is intended to be applied repeatedly. For example one could harvest
a sequence of the model predictions as follows:

    def posteriors(f,y):
        s = {}       
        x = list()
        for yi in y: 
            xi, xi_std, s = f(yi,s)
            x.append(xi)
        return x
 
Notice the use of s={} on first invocation. This is also important: 
- The caller should provide *a* pertaining to k-steps ahead, not the contemporaneous 'a'.  

### Packages incorporated: 

Not enough yet, as I got distracted by [HumpDay](https://github.com/microprediction/humpday) and the seeming paucity of fbprophet ([post](https://www.linkedin.com/posts/petercotton_is-facebooks-prophet-the-time-series-messiah-activity-6767451190679748608-ftGE)). However we're picking up speed and some functionality is drawn from:

  - fbprophet, 
  - pydlm, 
  - pmdarima,

and more. We are working down the [listing of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) and adding some home-grown approaches as well. 

   
### Skater "e" argument ("expiry")

The use of *e* is a fairly *weak* convention that many skaters ignore. In theory, a large expiry *e* can be used as a hint to the callee that
 there is time enough to do a 'fit', which we might define as anything taking longer than the usual function invocation.
 However, this is between the caller and it's priest really - or its prophet. Some skaters, such
 as the prophet skater, do a full 'fit' every invocation so this is meaningless. Other skaters
  no separate notion of 'fit' versus 'update' because everything is incremental. 
   

### Return values

For each prediction horizon it returns
two numbers where the first can be *interpreted* as a point estimate (but need not be) and the second is *typically* suggestive
of a symmetric error std, or width. Morally, a skater *suggests* an affine transformation of the incoming data. 


          -> x     [float],    # A vector of point estimates, or anchor points, or theos
             x_std [float]     # A vector of "scale" quantities (such as a standard deviation of expected forecast errors) 
             s    Any,         # Posterior state, intended for safe keeping by the callee until the next invocation 
                       

In returning state, the likely intent is that the *caller* might carry the state from one invocation to the next, not the *callee*. This is arguably more convenient than having the predicting object maintain state, because the caller can "freeze" the state as they see fit, as 
when making conditional predictions. This also eyes lambda-based deployments and *encourages* tidy use of internal state - not that we succeed
 when calling down to statsmodels (but all the home grown models here use simple dictionaries, making serialization trivial). See [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) if this seems odd). 

### Skater hyper-parameters in the interval (0,1)
 
Morally hyper-parameters occupy the cube but operationallly, we use an unusual convention. All model hyper-parameters must be squished down into
 a *scalar* quantity *r* in (0,1). This step may seem unnatural, but is workable with some space-filling curve conventions. More on that below. 
  
  
### Skater Elo ratings and rankings

Ratings for time series models, including some widely used packages such as fbprophet, are produced separately for different horizons. Specifically, we create a different Elo rating for looking k=1 steps ahead versus k=13 steps ahead, say. A rating is produced for each k in the Fibonacci sequence. See [skater_elo_ratings/leaderboards](https://github.com/microprediction/timemachines-testing/tree/main/skater_elo_ratings/leaderboards) sub-directories. For example some good ways to predict univariate time series 8 steps in advance might be suggested by the rankings at [/leaderboards/univariate_008](https://github.com/microprediction/timemachines-testing/tree/main/skater_elo_ratings/leaderboards/univariate_008) but of course their are caveats. 
  
    
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
     - See discussion below    
    
    
Nothing here is put forward
   as *the right way* to write time series packages - more a way of exposing their functionality for comparisons. 
  If you are interested in design thoughts for time series maybe participate in this [thread](https://github.com/MaxBenChrist/awesome_time_series_in_python/issues/1). 

### A little more about hyper-parameters

The restriction that all hyper-parameters be represented as r in (0,1) seems harsh. To be slightly less harsh, we include some standard ways
to use (0,1)^2 or (0,1)^3 should that be preferable. Admittedly, this may still not be the most natural way to represent choices, but here
we are trying to give lots of different optimizers a run at the problem. Of course, if you want to optimize skaters by some other means
and want a flexible way to represents searches then use one of the optim packages directly (e.g. optuna is pretty flexible in this regard). On 
the other hand, if you want to benefit from the operational simplicity of r in (0,1) then...

- The default map *from_space* from (0,1)^3 or (0,1)^2->(0,1) uses interleaving of digits in the binary representations (after first scaling).
- The script [demo_param_ordering.py](https://github.com/microprediction/timemachines/blob/master/examples/hyper/demo_param_ordering.py) illustrates
the mapping from r in (0,1) to R^n demonstrating that the first coordinate will vary
more smoothly as we vary r than the second, and so on.  
- If you need dim>3 for hyper-parameters, you can always use *to_space* or *from_space* more than once. 
- There are some functions provided to help you squish your hyper-params into the hypercube. The script [demo_balanced_log_scale.py](https://github.com/microprediction/timemachines/blob/master/examples/hyper/demo_balanced_log_scale.py) illustrates a
quasi-logarithmic parameter mapping from r in (0,1) to R which you can take or leave. 

[![IMAGE ALT TEXT](https://i.imgur.com/4F1oHXR.png)](https://vimeo.com/497113737 "Parameter importance")
Click to see video
 
 
### Optimization of hyperparameters
 
UPDATE: I'm moving the optimizer part of this package into a standalone library [HumpDay](https://github.com/microprediction/humpday).   
 

### Contributing 
If you'd like to contribute to this standardizing and benchmarking effort, here are some ideas:

- See the [list of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) ranked by download popularity. 
- Think about the most important hyper-parameters and consider "warming up" the mapping (0,1)->hyper-params by testing on real data. There is a [tutorial](https://www.microprediction.com/python-3) on retrieving live data, or use the [real data](https://pypi.org/project/realdata/) package, if that's simpler.
- The [comparison of hyper-parameter optimization packages](https://www.microprediction.com/blog/optimize) might also be helpful.  

If you are the maintainer of a time series package, we'd love your feedback and if you take the time to submit a PR here that incorporates your library, do yourself a favor and also enable "supporting" on your repo. 

See also [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md)
