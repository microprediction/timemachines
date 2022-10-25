# Timemachines: Predict with One Line of Code 
View as [web page](https://microprediction.github.io/timemachines/)

### Online timeseries forecasting

Here y is a vector or scalar, and we want to predict it three steps in advance.

     for yi in y:
         xi, x_std, s = f(y=yi, s=s, k=3)


*This package is a collection of a hundred or so "f"s.* 

### Why the Name?

The functions *f*, which are called [skaters](https://microprediction.github.io/timemachines/skaters.html), *suggest* state machines

  $$
    f : (y_t, state; k) \mapsto ( \hat{y}_{t+k}, \sigma, posterior\ state)
  $$

where $\sigma$ estimates the standard error of the prediction. 


-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
