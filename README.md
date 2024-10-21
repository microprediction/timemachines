# timemachines ![simple](https://github.com/microprediction/timemachines/workflows/tests/badge.svg)![tsa](https://github.com/microprediction/timemachines/workflows/test-tsa/badge.svg)  ![darts](https://github.com/microprediction/timemachines/workflows/test-darts/badge.svg) ![greykite](https://github.com/microprediction/timemachines/workflows/test-greykite/badge.svg)  ![sktime](https://github.com/microprediction/timemachines/workflows/test-sktime/badge.svg) ![tbats](https://github.com/microprediction/timemachines/workflows/test-tbats/badge.svg) ![simdkalman](https://github.com/microprediction/timemachines/workflows/test-simdkalman/badge.svg) ![prophet](https://github.com/microprediction/timemachines/workflows/test-prophet/badge.svg) ![statsforecast](https://github.com/microprediction/timemachines/workflows/test-statsforecast/badge.svg)![orbit](https://github.com/microprediction/timemachines/workflows/test-orbit/badge.svg)  ![neuralprophet](https://github.com/microprediction/timemachines/workflows/test-neuralprophet/badge.svg) ![pmd](https://github.com/microprediction/timemachines/workflows/test-pmd/badge.svg) ![pydlm](https://github.com/microprediction/timemachines/workflows/test-pydlm/badge.svg) ![merlion](https://github.com/microprediction/timemachines/workflows/test-merlion/badge.svg) ![merlion-prophet](https://github.com/microprediction/timemachines/workflows/test-merlion-prophet/badge.svg) ![river](https://github.com/microprediction/timemachines/workflows/test-river/badge.svg) ![divinity](https://github.com/microprediction/timemachines/workflows/test-divinity/badge.svg)![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret-time_series/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Not currently supported in continuous integration:
![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret/badge.svg)
![successor](https://github.com/microprediction/timemachines/workflows/test-successor/badge.svg)


# Simple prediction functions ([documented](https://microprediction.github.io/timemachines/) and [assessed](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html))  
Because [why not](https://microprediction.github.io/timemachines/why) do things in **walk-forward incremental** fashion with **one line of code**? Here yt is a vector or scalar, and we want to predict yt (or its first coordinate if a vector) three steps in advance. 

     from timemachines.skaters.somepackage.somevariety import something as f
     for yt in y:
         xt, xt_std, s = f(y=yt, s=s, k=3)
         
This emits a k-vector xt of forecasts, and corresponding k-vector xt_std of estimated standard errors. See [skaters](https://microprediction.github.io/timemachines/skaters) for choices of somepackage, somevariety and something. You can also ensemble, compose, bootstrap and do other things with one line of code. 

See the [docs](https://microprediction.github.io/timemachines/). 

### Packages used

Skaters draw on functionality from [popular python time-series packages](https://www.microprediction.com/blog/popular-timeseries-packages) like [river](https://github.com/online-ml/river), [pydlm](https://github.com/wwrechard/pydlm), [tbats](https://github.com/intive-DataScience/tbats), [pmdarima](http://alkaline-ml.com/pmdarima/), [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html), [neuralprophet](https://neuralprophet.com/), Facebook [Prophet](https://facebook.github.io/prophet/), 
   Uber's [orbit](https://eng.uber.com/orbit/), Facebook's [greykite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library) and more. See the [docs](https://microprediction.github.io/timemachines/).

## What's a "skater"?
More abstractly:

$$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
$$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. 

If you prefer an legitimate (i.e. stateful) state machine, see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 


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

### Contributions and capstone projects
    
- See [TFRO.md](https://github.com/microprediction/monteprediction/blob/main/TFRO.md) 
- See [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md) and [good first issues](https://github.com/microprediction/timemachines/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). 
- See the suggested steps for a [capstone project](https://microprediction.github.io/timemachines/capstone.html). 

### Getting live help

- [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md).
- Office hours [here](https://github.com/microprediction/meet). 
- Learn how to deploy some of these models and try to win the [daily $125 prize](https://www.microprediction.com/competitions/daily).

### Install [instructions](https://github.com/microprediction/timemachines/blob/main/INSTALL.md)

Oh what a mess the Python timeseries ecosystem is. So packages are not installed by default. See the methodical [install instructions](https://github.com/microprediction/timemachines/blob/main/INSTALL.md) and be incremental for best results. The infamous [xkcd cartoon](https://xkcd.com/1987/) really does describe the alternative quite well. 


![](https://i.imgur.com/elu5muO.png)
  
    
### Cite 

Thanks

        @electronic{cottontimemachines,
            title = {{Timemachines: A Python Package for Creating and Assessing Autonomous Time-Series Prediction Algorithms}},
            year = {2021},
            author = {Peter Cotton},
            url = {https://github.com/microprediction/timemachines}
        }

or something [here](https://github.com/microprediction/microprediction/blob/master/CITE.md). 
