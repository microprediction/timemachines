# Timemachines: Predict with One Line of Code 
View as [web page](https://microprediction.github.io/timemachines/)

### What's in here?
Implemntation of "skaters" which are functions *suggesting* state machines for prediction

  $$
    f : (y_t, state; k) \mapsto ( \hat{y}_{t+k}, \sigma, posterior\ state)
  $$

where $\sigma$ estimates the standard error of the prediction. (If you want an actual state machine, 
see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1.) 


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
