# timemachines ![simple](https://github.com/microprediction/timemachines/workflows/tests/badge.svg)![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret/badge.svg)![tsa](https://github.com/microprediction/timemachines/workflows/test-tsa/badge.svg) ![successor](https://github.com/microprediction/timemachines/workflows/test-successor/badge.svg) ![darts](https://github.com/microprediction/timemachines/workflows/test-darts/badge.svg) ![greykite](https://github.com/microprediction/timemachines/workflows/test-greykite/badge.svg)  ![sktime](https://github.com/microprediction/timemachines/workflows/test-sktime/badge.svg) ![tbats](https://github.com/microprediction/timemachines/workflows/test-tbats/badge.svg) ![simdkalman](https://github.com/microprediction/timemachines/workflows/test-simdkalman/badge.svg) ![prophet](https://github.com/microprediction/timemachines/workflows/test-prophet/badge.svg) ![orbit](https://github.com/microprediction/timemachines/workflows/test-orbit/badge.svg)  ![neuralprophet](https://github.com/microprediction/timemachines/workflows/test-neuralprophet/badge.svg) ![pmd](https://github.com/microprediction/timemachines/workflows/test-pmd/badge.svg) ![pydlm](https://github.com/microprediction/timemachines/workflows/test-pydlm/badge.svg) ![merlion](https://github.com/microprediction/timemachines/workflows/test-merlion/badge.svg) ![merlion-prophet](https://github.com/microprediction/timemachines/workflows/test-merlion-prophet/badge.svg) ![river](https://github.com/microprediction/timemachines/workflows/test-river/badge.svg) ![divinity](https://github.com/microprediction/timemachines/workflows/test-divinity/badge.svg)![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret-time_series/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Simple to use forecasting functions assigned [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)
Here y is a vector or scalar, and we want to predict it three steps in advance.

     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)

This package is a collection of a hundred or so "f"s, nothing more.  

### Simple uses of this package:

1. Use some of the functionality of a [subset](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/pypi.py) of the [popular python time-series packages](https://www.microprediction.com/blog/popular-timeseries-packages) like [river](https://github.com/online-ml/river), [pydlm](https://github.com/wwrechard/pydlm), [tbats](https://github.com/intive-DataScience/tbats), [pmdarima](http://alkaline-ml.com/pmdarima/), [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html), [neuralprophet](https://neuralprophet.com/), Facebook [Prophet](https://facebook.github.io/prophet/), 
   Uber's [orbit](https://eng.uber.com/orbit/), Facebook's [greykite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library) and more with one line of code. Or use home-spun methods like [thinking_fast_and_slow](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/thinking.py) that you'll only find here. 
2. Peruse [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) or use them [programatically](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/recommendations/suggestions.py). There's also a recommendation [colab notebook](https://github.com/microprediction/timeseries-elo-ratings/blob/main/time_series_recommendations.ipynb) you can open and run. And you might consider the use of [forever functions](https://www.microprediction.com/blog/forever) that get better over time without your doing anything.

### Hey do you reckon latex works in markdown on GitHub now?
This is a skater

  $$
     (current\ data, state) \mapsto (mean, cov, posterior\ state)
  $$
Oh my life just changed (see all the [happy people](https://www.linkedin.com/posts/petercotton_math-support-in-markdown-the-github-blog-activity-6933213212783902721-QM7H?utm_source=linkedin_share&utm_medium=member_desktop_web)!)

### More advanced uses of this package:

3. Make your own autonomous algorithms and watch them compete. See the [daily $125 prize](https://www.microprediction.com/competitions/daily) and open this [notebook](https://github.com/microprediction/microprediction/blob/master/submission_examples_die/first_submission.ipynb) to understand the rudimentary mechanics of submitting distributions. Any skaters in this package can be turned into a "crawler" pretty easily, as demonstrated in the [stream skater examples](https://github.com/microprediction/microprediction/tree/master/crawler_skater_examples). 
4. Use [stacking](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools/ensembling) to create better skaters. One can, for example, draw on a large inventory of [online portfolio managers](https://github.com/microprediction/precise/blob/main/LISTING_OF_MANAGERS.md) in the *precise* package to assign positive weights to a collection of skaters based on their model residuals. If you have the computation time, you can draw on all the latest advances in portfolio theory.  
5. Use hyper-parameter [tuning](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/tuning/README.md) and turn "almost" autonomous algorithms, or combinations of the same, into fully autonomous algorithms using just about any global optimizer you can think of via the humpday package. At present there are about [one hundred derivative-free methods](https://microprediction.github.io/optimizer-elo-ratings/html_leaderboards/overall.html) including choices from [Ax-Platform](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/axcube.py), [bayesian-optimization](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/bayesoptcube.py), [DLib](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/dlibcube.py), [HyperOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/hyperoptcube.py), [NeverGrad](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/nevergradcube.py), [Optuna](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/optunacube.py), [Platypus](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/platypuscube.py), [PyMoo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pymoocube.py), [PySOT](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/pysotcube.py), Scipy [classic](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/scipycube.py) and [shgo](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/shgocube.py), [Skopt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/skoptcube.py),
[nlopt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/nloptcube.py), [Py-Bobyaq](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/bobyqacube.py), and
[UltraOpt](https://github.com/microprediction/humpday/blob/main/humpday/optimizers/ultraoptcube.py). 
6. Use [composition](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/composition/residualshypocratic.py) to chase residuals (like boosting). Determine whether skaters here help predict your proprietary in-model residuals.  
7. Write your next paper and easily benchmark your work, using [live data](https://www.microprediction.org/browse_streams.html). Or write an [Empirical Article That Isn't Immediately Stale](https://medium.com/geekculture/an-empirical-article-that-wasnt-immediately-stale-720abfb4678f).  

### Contribute
    
See [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md).  Also [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md). 

### Slack / Google Meets
See the Slack invite on my user page [here](https://github.com/microprediction). 

### [Daily $125 prize](https://www.microprediction.com/competitions/daily)
Figured that might be worth repeating. 

<img src="https://github.com/microprediction/timemachines/blob/main/images/money.png" alt="Money" style="width:500px">

Who says contributing to open-source is thankless? 

# [Installation](https://github.com/microprediction/timemachines/blob/main/INSTALL.md)

See the methodical [install instructions](https://github.com/microprediction/timemachines/blob/main/INSTALL.md) and be incremental for best results. This [xkcd cartoon](https://xkcd.com/1987/) describes the alternative quite well, especially in the time-series ecosystem. 

 
## Examples

See [examples](https://github.com/microprediction/timemachines/tree/main/examples) 

# [Quick start](https://github.com/microprediction/timemachines/tree/main/examples/basic_usage) 

Again, this package is *merely* a collection of "skater" functions. "Skater" is a nmemonic for the arguments, although you might need only 1 or 2. The script [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py) illustrates the usage pattern. Like so:

    from timemachines.skaters.simple.thinking import thinking_slow_and_fast 
    import numpy as np
    y = np.cumsum(np.random.randn(1000))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = thinking_slow_and_fast(y=yi, s=s, k=3)
        x.append(xi)
     
There's more in [examples/basic_usage](https://github.com/microprediction/timemachines/tree/main/examples/basic_usage).
  
  
![](https://i.imgur.com/elu5muO.png)
  
# Why do it this way?
Oh here comes the justification. 

   1. **Simple k-step ahead forecasts in functional style** I mean that's obvious. There are no "models" here requiring setup or estimation, only stateless functions as noted:
       
          x, x_std, s = f(y,s,k)
         
and if you need, 

          x, x_std, s = f(y,s,k,a,t,e,r)
       
The s-k-a-t-e-r functions take on the responsibility of incremental estimation, so you don't have to. Some skaters are computationally efficient in this respect, whereas others are drawn from traditional packages intended for batch/offline work, and can be quite slow when called repeatedly. But they are here because it is necessary to compare the accuracy of fast and slow algorithms, even if the latter might not suit your production volumetrics. 

   2. **Ongoing, incremental, empirical evaluation**. Again, see the [leaderboards](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) produced by
    the accompanying repository [timeseries-elo-ratings](https://github.com/microprediction/timeseries-elo-ratings). Assessment is always out of sample and uses *live*, constantly updating real-world data 
     from [microprediction.org](https://www.microprediction.org/browse_streams.html).   
    
   3. **Terse stacking, ensembling, boosting and other combining** of models. The function form makes it easy to do this, usually with one or two lines of code (again, see [thinking.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/thinking.py) for an illustration, 
   or [prophetskaterscomposed.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/proph/prophskaterscomposed.py)).

   4. **Simplified deployment**. There is no state, other that that explicitly returned to the caller. For skaters relying only on the timemachines and river packages (the fast ones), the state is a pure Python dictionary trivially converted to JSON and back (for instance in a web application). See the [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) for a little more discussion.   

So there you have it. No classes. Few dataframes. Hopefully little ceremony.   

  
# Skating 

By the way, the name *timemachines* is chosen because the skater functions *suggest* state machines for sequential assimilation of observations (as a data point arrives, 
    forecasts for 1,2,...,k steps ahead, with corresponding standard deviations are emitted). However unlike state machines that save state themselves, here the *caller* is expected to maintain state from one 
    invocation (data point) to the next. See the [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) if this seems odd. 

So, that's what's going on with the *s*. As for the k-a-t-e-r:  

      x, w, s = f(   y:Union[float,[float]],             # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:[float]=None,                          # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for hype(r)-pa(r)amete(r)s) 

Since only one *y* arrives at a time, it is up to you to harvest a sequence of the model predictions, if you wish to, as follows:

    def posteriors(f,y):
        s = {}       
        x = list()
        for yi in y: 
            xi, xi_std, s = f(yi,s)
            x.append(xi)
        return x

Though you *could* use the prominently positioned [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py) utilities for processing full histories.  
 
### Skater "y" argument

A skater function *f* takes a vector *y*, where:

    - The quantity to be predicted (target) is y[0] and,
    - There may be other, simultaneously observed variables y[1:] deemed helpful in predicting y[0].

### Skater "s" argument
 
The internal state of the skater, intended to summarize everything the skater needs to know from the past. It is entirely up to the creator of the skater to devise a sensible scheme for s, and it is advised that this be a dictionary that is easily dumped to JSON, although it is impossible to enforce this constraint when incorporating third party packages. 

### Skater "k" argument 

The integer k argument determines the length of the vector of predictions (and also their standard deviations) that will be returned. 

Typically if you really care about k=1 step ahead prediction then you should specify k=1. You *could* specify k=5 or whatever and take the first element of the 5-vector returned, but many skaters might not ensure this is the same as when k=1 is specified (for computational reasons some interpolation might be applied, for instance). 

### Skater "a" argument 

A vector of known-in-advance variables. For instance a day of week. 

You can also use the "a" argument for conditional prediction. This is a matter of interpretation. For instance, you might ask for two predictions, one with a=0 and one with a=1 where this represents rain or something. 

It must be said that at present, most skaters will ignore the "a" argument entirely. 

Also, there are at present no universal conventions for values taken by "a". It is entirely up to the skater to infer this. 

### Skater "t" argument 

Epoch time of the observation. 

Again, many skaters will choose to ignore this input and instead presume that data is regularly sampled. 

### Skater "e" argument ("expiry")

A loose convention but:

       e < 0    -  Tells skater that it should update the state but the actual emitted result won't be used. 
       e = 0    -  Tells skater that the result will matter, so be sure to compute it. 
       e > 0    -  Tells the skater that it has plenty of time, so maybe performing a periodic "fit" is in order, if that's something it does. 
       
This can be very useful for testing, since we can set e<0 during burn-in. 
   
### Skater "r" argument (stands for "hype(r) pa(r)amete(r)s for p(r)e-skaters only)

A real skater doesn't have any hyper-parameters. It's the job of the designer to make it fully autonomous. The small concession made here is the notion of a pre-skater: one with a single float hyperparameter in the closed interval \[0,1\]. Pre-skaters squish all tunable parameters into this interval. That's a bit tricky, so some rudimentary conventions and space-filling functions are provided. See [tuning](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools/tuning) and longer discussion below. 

## Return values
All skater functions return two vectors and the posterior state dictionary. 

      1. The first set of *k* numbers can be *interpreted* as a point estimate (but need not be)
      2. The second is *typically* suggestive of a symmetric error std, or width. However a broader interpretation is possible wherein a skater *suggests* a useful affine transformation of the incoming data and nothing more.  
      3. The third returned value is state, and the skater expects to receive it again on the next call.


          -> x     [float],    # A vector of point estimates, or anchor points, or theos
             x_std [float]     # A vector of "scale" quantities (such as a standard deviation of expected forecast errors) 
             s    Any,         # Posterior state, intended for safe keeping by the callee until the next invocation 
                       
For many skaters the x_std is, as is suggested, indicative of one standard deivation. 

     x, x_std, x = f( .... )   # skater
     x_up = [ xi+xstdi for xi,xstdi in zip(x,xstd) ]
     x_dn = [ xi-xstdi for xi,xstdi in zip(x,xstd) ]
     
then very roughly the k'th next value should, with 5 out of 6 times, below the k'th entry in x_up 
There isn't any capability to indicate three numbers (e.g. for asymmetric conf intervals around the mean).  

In returning state, the intent is that the *caller* might carry the state from one invocation to the next verbatim. This is arguably more convenient than having the predicting object maintain state, because the caller can "freeze" the state as they see fit, as 
when making conditional predictions. This also eyes lambda-based deployments and *encourages* tidy use of internal state - not that we succeed
 when calling down to statsmodels (though prophet, and others including the home grown models use simple dictionaries, making serialization trivial). 
    
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

- Expiry
     - The "e" parameter serves as a performance hack  
     - e<0 tells the skater that the output won't be used, so worry only about state updating
     - e=0 tells the skater that there isn't much time, so this particular invocation isn't the best time to do a periodic "fit" exercise. 

- Hyper-Parameter space (for "pre-skaters")
     - A float *r* in (0,1). 
     - This package provides functions *to_space* and *from_space*, for expanding to R^n using space filling curves, so that the callee's (hyper) parameter optimization can still exploit geometry, if it wants to.   
     - See [tuning](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools/tuning)
     
See [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) or file an issue if anything offends you greatly. 
 
### Aside: more on the e argument ...
Now back to *e* again. 

Some skaters are so fast that a separate notion of 'fit' versus 'update' is irrelevant. Other skaters will periodically fit whether or not e>0 is passed. 
For some, there is even more graduated performance so *e* could be interpreted as "number of seconds allowed". To be safe the tests often pass an e sequence like the following: 

     -1, -1, -1, ... -1 1000 1000 1000 1000 1000 ...
     
because it wants to allow the skaters to receive some history before they are evaluated. Often you just wan the skater to shove that in a buffer and not compute anything, if that is its style, becaue waiting for Facebook prophet to fit itself 500 times is a bit like waiting for the second coming of Christ. 
    
## Cite 

Thanks

        @electronic{cottontimemachines,
            title = {{Timemachines: A Python Package for Creating and Assessing Autonomous Time-Series Prediction Algorithms}},
            year = {2021},
            author = {Peter Cotton},
            url = {https://github.com/microprediction/timemachines}
        }

or something [here](https://github.com/microprediction/microprediction/blob/master/CITE.md). 
