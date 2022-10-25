# timemachines ![simple](https://github.com/microprediction/timemachines/workflows/tests/badge.svg)![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret/badge.svg)![tsa](https://github.com/microprediction/timemachines/workflows/test-tsa/badge.svg) ![successor](https://github.com/microprediction/timemachines/workflows/test-successor/badge.svg) ![darts](https://github.com/microprediction/timemachines/workflows/test-darts/badge.svg) ![greykite](https://github.com/microprediction/timemachines/workflows/test-greykite/badge.svg)  ![sktime](https://github.com/microprediction/timemachines/workflows/test-sktime/badge.svg) ![tbats](https://github.com/microprediction/timemachines/workflows/test-tbats/badge.svg) ![simdkalman](https://github.com/microprediction/timemachines/workflows/test-simdkalman/badge.svg) ![prophet](https://github.com/microprediction/timemachines/workflows/test-prophet/badge.svg) ![statsforecast](https://github.com/microprediction/timemachines/workflows/test-statsforecast/badge.svg)![orbit](https://github.com/microprediction/timemachines/workflows/test-orbit/badge.svg)  ![neuralprophet](https://github.com/microprediction/timemachines/workflows/test-neuralprophet/badge.svg) ![pmd](https://github.com/microprediction/timemachines/workflows/test-pmd/badge.svg) ![pydlm](https://github.com/microprediction/timemachines/workflows/test-pydlm/badge.svg) ![merlion](https://github.com/microprediction/timemachines/workflows/test-merlion/badge.svg) ![merlion-prophet](https://github.com/microprediction/timemachines/workflows/test-merlion-prophet/badge.svg) ![river](https://github.com/microprediction/timemachines/workflows/test-river/badge.svg) ![divinity](https://github.com/microprediction/timemachines/workflows/test-divinity/badge.svg)![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret-time_series/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Simple [documented](https://microprediction.github.io/timemachines/) timeseries prediction functions assigned [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)
Here y is a vector or scalar, and we want to predict it three k=3 steps in advance.

     from timemachines.skaters.somepackage.somevariety import something as f
     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)

See [skaters](https://microprediction.github.io/timemachines/skaters) for choices of somepackage, somevariety and something. 
This package is a collection of a hundred or so "f"s, for people who want prediction without ceremoney. There are also utilities for ensembling, composing and assessing prediction functions in ways that cannot be p-hacked.  

More abstractly:

$$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
$$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. If you prefer an legit (stateful) state machine, see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 


### Skater function conventions

See [docs/interface](https://microprediction.github.io/timemachines/interface) for description of skater inputs and outputs. Briefly:

      x, w, s = f(   y:Union[float,[float]],             # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:[float]=None,                          # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for hype(r)-pa(r)amete(r)s) 


### Uses of this package

See [uses](https://microprediction.github.io/timemachines/uses) in the docs. 


### Contribute
    
See [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md).  

### FAQ

[FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md). 

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

  
    
## Cite 

Thanks

        @electronic{cottontimemachines,
            title = {{Timemachines: A Python Package for Creating and Assessing Autonomous Time-Series Prediction Algorithms}},
            year = {2021},
            author = {Peter Cotton},
            url = {https://github.com/microprediction/timemachines}
        }

or something [here](https://github.com/microprediction/microprediction/blob/master/CITE.md). 
