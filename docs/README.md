# Timemachines: Predict with One Line of Code  ([install](https://github.com/microprediction/timemachines/blob/main/INSTALL.md))

View as [web page](https://microprediction.github.io/timemachines/)

### Online (incremental) k-step ahead prediction
Implementations of "skaters": stateless functions *suggesting* state machines that process one observation vector (or scalar) at a time and emit k-step ahead prediction vectors:

  $$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
  $$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. If you prefer an legit (stateful) state machine, see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 


### Using a skater (see [list](https://microprediction.github.io/timemachines/skaters.html) of them)

     from timemachines.skaters.somewhere import something as f 
     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)

Here *yi* is a scalar or vector. Again, see [skaters](https://microprediction.github.io/timemachines/skaters.html) for more detail on the
*somewhere* and *something*. 

### Other ways to [use](https://microprediction.github.io/timemachines/uses) this package
Beyond the fact that there's a [hundred or so](https://microprediction.github.io/timemachines/skaters.html) choices of "f", that allow you to use
functionality from most popular timeseries packages with one line of code, you can also: 

- Compose, ensemble, wiggle, assess using [skatertools](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools).
- Glean a vague idea of what works from the sister [repo](https://github.com/microprediction/timeseries-elo-ratings) which creates [timeseries Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/residual-k_013.html) for the skaters. 

See also [uses](https://microprediction.github.io/timemachines/uses).  

### What's not in here?
That [list](https://www.microprediction.com/blog/popular-timeseries-packages) is long. Please [help add more](https://github.com/microprediction/timemachines/issues?q=is%3Aissue+is%3Aopen+label%3A%22create+colab+example%22)! The scope is mostly
limited to univariate prediction, at present. 

### Contribute
    
See [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md).  

### FAQ

[FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md). 

### Human help
See [google meets](https://microprediction.github.io/microprediction/meet.html)

### [Daily $125 prize](https://www.microprediction.com/competitions/daily)
Who says contributing to open-source is thankless? 

# [Install](https://github.com/microprediction/timemachines/blob/main/INSTALL.md)

See the methodical [install instructions](https://github.com/microprediction/timemachines/blob/main/INSTALL.md) and be incremental for best results. This [xkcd cartoon](https://xkcd.com/1987/) describes the alternative quite well, especially in the time-series ecosystem. 

## Examples

See [examples](https://github.com/microprediction/timemachines/tree/main/examples) 

## Book
There's a [book website](https://microprediction.github.io/building_an_open_ai_network/) motivating autonomous critters, but it ain't an instruction manual, sorry. 

-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
