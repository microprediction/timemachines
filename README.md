# timemachines ![simple](https://github.com/microprediction/timemachines/workflows/tests/badge.svg)![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret/badge.svg)![tsa](https://github.com/microprediction/timemachines/workflows/test-tsa/badge.svg) ![successor](https://github.com/microprediction/timemachines/workflows/test-successor/badge.svg) ![darts](https://github.com/microprediction/timemachines/workflows/test-darts/badge.svg) ![greykite](https://github.com/microprediction/timemachines/workflows/test-greykite/badge.svg)  ![sktime](https://github.com/microprediction/timemachines/workflows/test-sktime/badge.svg) ![tbats](https://github.com/microprediction/timemachines/workflows/test-tbats/badge.svg) ![simdkalman](https://github.com/microprediction/timemachines/workflows/test-simdkalman/badge.svg) ![prophet](https://github.com/microprediction/timemachines/workflows/test-prophet/badge.svg) ![statsforecast](https://github.com/microprediction/timemachines/workflows/test-statsforecast/badge.svg)![orbit](https://github.com/microprediction/timemachines/workflows/test-orbit/badge.svg)  ![neuralprophet](https://github.com/microprediction/timemachines/workflows/test-neuralprophet/badge.svg) ![pmd](https://github.com/microprediction/timemachines/workflows/test-pmd/badge.svg) ![pydlm](https://github.com/microprediction/timemachines/workflows/test-pydlm/badge.svg) ![merlion](https://github.com/microprediction/timemachines/workflows/test-merlion/badge.svg) ![merlion-prophet](https://github.com/microprediction/timemachines/workflows/test-merlion-prophet/badge.svg) ![river](https://github.com/microprediction/timemachines/workflows/test-river/badge.svg) ![divinity](https://github.com/microprediction/timemachines/workflows/test-divinity/badge.svg)![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret-time_series/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Simple [documented](https://microprediction.github.io/timemachines/) timeseries prediction functions assigned [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)
Because [why not](https://microprediction.github.io/timemachines/why) do things with one line of code? Here y is a vector or scalar, and we want to predict it three k=3 steps in advance.

     from timemachines.skaters.somepackage.somevariety import something as f
     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)

See [skaters](https://microprediction.github.io/timemachines/skaters) for choices of somepackage, somevariety and something. 
This package is a collection of a hundred or so "f"s, for people who want one-line prediction sans ceremony or state. See [uses](https://microprediction.github.io/timemachines/uses) in the docs. There are utilities for ensembling, [residual chasing](https://microprediction.github.io/timemachines/composition) and assessing prediction functions in ways that cannot be p-hacked. And the goal here is to include some that do a reasonable job of providing fully autonomous prediction across [diverse real-world timeseries](https://www.microprediction.org/browse_streams.html), thus providing a cheap lunch.   

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


### Contribute
    
See [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md).  

### Help

- [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md). 
- See the Slack invite on my user page [here](https://github.com/microprediction/slack). 
- Office hours [here](https://github.com/microprediction/meet). 

### [Daily $125 prize](https://www.microprediction.com/competitions/daily)
Figured that might be worth a mention.  

<img src="https://github.com/microprediction/timemachines/blob/main/images/money.png" alt="Money" style="width:500px">

Who says contributing to open-source is thankless? 

# [Installation](https://github.com/microprediction/timemachines/blob/main/INSTALL.md)

See the methodical [install instructions](https://github.com/microprediction/timemachines/blob/main/INSTALL.md) and be incremental for best results. This [xkcd cartoon](https://xkcd.com/1987/) describes the alternative quite well, especially in the time-series ecosystem. 
 
## Examples of basic usage
See [examples/basic_usage](https://github.com/microprediction/timemachines/tree/main/examples/basic_usage) and [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py) illustrating the usage pattern. To wit:

    from timemachines.skaters.simple.thinking import thinking_slow_and_fast 
    import numpy as np
    y = np.cumsum(np.random.randn(1000))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = thinking_slow_and_fast(y=yi, s=s, k=3)
        x.append(xi)
     

![](https://i.imgur.com/elu5muO.png)
  
    
## Cite 

Thanks

        @electronic{cottontimemachines,
            title = {{Timemachines: A Python Package for Creating and Assessing Autonomous Time-Series Prediction Algorithms}},
            year = {2021},
            author = {Peter Cotton},
            url = {https://github.com/microprediction/timemachines}
        }

or something [here](https://github.com/microprediction/microprediction/blob/master/CITE.md). 
