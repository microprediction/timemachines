# timemachines ([docs](https://microprediction.github.io/timemachines/)) ![simple](https://github.com/microprediction/timemachines/workflows/tests/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Univariate prediction functions from diverse packages supported in a simple stateless pure function syntax, mosty for benchmarking and application-specific selection purposes. See [basic usage](https://github.com/microprediction/timemachines/blob/main/examples/basic_usage/run_skater.py). Briefly: if `yt` is a list of floats we can feed them one at a time to a skater like so:

     from timemachines.skaters.somepackage.somevariety import something as f
     for yt in y:
         xt, xt_std, s = f(y=yt, s=s, k=3)
         
This emits a k-vector `xt` of forecasts, and corresponding k-vector `xt_std` of estimated standard errors, and the posterior state `s` needed for the next call. See [skaters](https://microprediction.github.io/timemachines/skaters) for choices of `somepackage`, `somevariety` and `something`. You can also ensemble, compose, bootstrap and do other things with one line of code. The `f` is called a `skater`. These are ([documented](https://microprediction.github.io/timemachines/) and [assessed](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)).  See [why](https://microprediction.github.io/timemachines/why) for motivation for doing things in **walk-forward incremental** functional fashion with **one line of code**. 

Test for the following packages are expected to be working, and their use in this context reliable: 
- ![tsa](https://github.com/microprediction/timemachines/workflows/test-tsa/badge.svg)    Statsmodels 
- ![darts](https://github.com/microprediction/timemachines/workflows/test-darts/badge.svg)
- ![simdkalman](https://github.com/microprediction/timemachines/workflows/test-simdkalman/badge.svg)
- ![prophet](https://github.com/microprediction/timemachines/workflows/test-prophet/badge.svg)
- ![pydlm](https://github.com/microprediction/timemachines/workflows/test-pydlm/badge.svg)
- ![merlion](https://github.com/microprediction/timemachines/workflows/test-merlion/badge.svg)
- ![merlion-prophet](https://github.com/microprediction/timemachines/workflows/test-merlion-prophet/badge.svg)
- ![river](https://github.com/microprediction/timemachines/workflows/test-river/badge.svg)

Test for the following packages are not necessarily expected to be working:
- ![greykite](https://github.com/microprediction/timemachines/workflows/test-greykite/badge.svg) 
- ![sktime](https://github.com/microprediction/timemachines/workflows/test-sktime/badge.svg)
- ![tbats](https://github.com/microprediction/timemachines/workflows/test-tbats/badge.svg)
- ![statsforecast](https://github.com/microprediction/timemachines/workflows/test-statsforecast/badge.svg)
- ![orbit](https://github.com/microprediction/timemachines/workflows/test-orbit/badge.svg)
- ![neuralprophet](https://github.com/microprediction/timemachines/workflows/test-neuralprophet/badge.svg)
- ![pmd](https://github.com/microprediction/timemachines/workflows/test-pmd/badge.svg)
- ![divinity](https://github.com/microprediction/timemachines/workflows/test-divinity/badge.svg)
- ![pycaret](https://github.com/microprediction/timemachines/workflows/test-pycaret-time_series/badge.svg)
- ![successor](https://github.com/microprediction/timemachines/workflows/test-successor/badge.svg)

See the [docs](https://microprediction.github.io/timemachines/). 

## What's a "skater"?
More abstractly:

$$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
$$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. If you prefer an legitimate (i.e. stateful) state machine, that's fine but see see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1 for how to quickly roll this yourself. 

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

### Contributing
    
- See [TFRO.md](https://github.com/microprediction/monteprediction/blob/main/TFRO.md) 
- See [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md) and [good first issues](https://github.com/microprediction/timemachines/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). 
- See the suggested steps for a [capstone project](https://microprediction.github.io/timemachines/capstone.html). 

### Getting live help

- Office hours are mentioned [here](https://github.com/microprediction). 
- Discord channle is mentioned [here](https://github.com/microprediction). 

See also [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md).

### Install [instructions](https://github.com/microprediction/timemachines/blob/main/INSTALL.md)

Oh what a mess the Python timeseries ecosystem is. So packages are not installed by default. See the methodical [install instructions](https://github.com/microprediction/timemachines/blob/main/INSTALL.md) and be incremental for best results. The infamous [xkcd cartoon](https://xkcd.com/1987/) really does describe the alternative quite well. 


![](https://i.imgur.com/elu5muO.png)

### Links to packages used

Skaters draw on functionality from [popular python time-series packages](https://www.microprediction.com/blog/popular-timeseries-packages) like [river](https://github.com/online-ml/river), [pydlm](https://github.com/wwrechard/pydlm), [tbats](https://github.com/intive-DataScience/tbats), [pmdarima](http://alkaline-ml.com/pmdarima/), [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html), [neuralprophet](https://neuralprophet.com/), Facebook [Prophet](https://facebook.github.io/prophet/), 
   Uber's [orbit](https://eng.uber.com/orbit/), Facebook's [greykite](https://engineering.linkedin.com/blog/2021/greykite--a-flexible--intuitive--and-fast-forecasting-library) and more. See the [docs](https://microprediction.github.io/timemachines/).
    
### Cite 

Thanks

        @electronic{cottontimemachines,
            title = {{Timemachines: A Python Package for Creating and Assessing Autonomous Time-Series Prediction Algorithms}},
            year = {2021},
            author = {Peter Cotton},
            url = {https://github.com/microprediction/timemachines}
        }

or something [here](https://github.com/microprediction/microprediction/blob/master/CITE.md). 
