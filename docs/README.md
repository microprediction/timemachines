# Timemachines: Predict with One Line of Code 
View as [web page](https://microprediction.github.io/timemachines/)

### Online timeseries forecasting using [skaters](https://microprediction.github.io/timemachines/skaters.html)

     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)

Here *yi* is a scalar or vector. See [skaters](https://microprediction.github.io/timemachines/skaters.html).  

### What's in here? 

This package is a collection of a hundred or so "f"s, drawn from [popular timeseries packages](https://microprediction.github.io/timemachines/skaters.html) and utilities for making combinations of the same. 

### What's not in here?
That [list](https://www.microprediction.com/blog/popular-timeseries-packages) is long. Please [help add more](https://github.com/microprediction/timemachines/issues?q=is%3Aissue+is%3Aopen+label%3A%22create+colab+example%22)!   

### Why the name "timemachines"? 
Skaters *suggest* state machines

  $$
    f : (y_t, state; k) \mapsto ( \hat{y}_{t+k}, \sigma, posterior\ state)
  $$

where $\sigma$ estimates the standard error of the prediction. If you want an actual state machine, 
see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 


-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
