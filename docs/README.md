# Timemachines: Predict with One Line of Code 
View as [web page](https://microprediction.github.io/timemachines/)

### Online (incremental) k-step ahead prediction
Implementations of "skaters": stateless functions *suggesting* state machines that process one observation vector (or scalar) at a time and emit k-step ahead prediction vectors:

  $$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
  $$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. If you prefer an legit (stateful) state machine, see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 


### Using a skater (see [list](https://microprediction.github.io/timemachines/skaters.html) of them)

     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)

Here *yi* is a scalar or vector. Again, see [skaters](https://microprediction.github.io/timemachines/skaters.html) for more detail and
ways to find and import ones you like. 

### Ways to [use](https://microprediction.github.io/timemachines/uses) this package
This package contains:

- A hundred or so "f"s, drawn from [popular timeseries packages](https://microprediction.github.io/timemachines/skaters.html) 
- Utilities for combining, assessing, et cetera. See [uses](https://microprediction.github.io/timemachines/uses).  
- A sister repo which creates [timeseries Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/residual-k_013.html). 

### What's not in here?
That [list](https://www.microprediction.com/blog/popular-timeseries-packages) is long. Please [help add more](https://github.com/microprediction/timemachines/issues?q=is%3Aissue+is%3Aopen+label%3A%22create+colab+example%22)! The scope is mostly
limited to univariate prediction, at present. 




-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
