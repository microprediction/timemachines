# Timemachines: Predict with One Line of Code 
View as [web page](https://microprediction.github.io/timemachines/)

### Online (incremental) k-step ahead prediction
Implementations of "skaters": functions *suggesting* state machines emitting k-step ahead prediction vectors:

  $$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
  $$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. 

Btw if you want an actual state machine, see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 


### Using a [skater](https://microprediction.github.io/timemachines/skaters.html)
is easy:

     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)

Here *yi* is a scalar or vector. See [skaters](https://microprediction.github.io/timemachines/skaters.html) for how
to find and import "skater" f. 

### What's it [useful]((https://microprediction.github.io/timemachines/uses) for?
This package contains:

- A hundred or so "f"s, drawn from [popular timeseries packages](https://microprediction.github.io/timemachines/skaters.html) 
- Utilities for combining, assessing, et cetera. See [uses](https://microprediction.github.io/timemachines/uses).  

### What's not in here?
That [list](https://www.microprediction.com/blog/popular-timeseries-packages) is long. Please [help add more](https://github.com/microprediction/timemachines/issues?q=is%3Aissue+is%3Aopen+label%3A%22create+colab+example%22)! The scope is mostly
limited to univariate prediction, at present. 




-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
