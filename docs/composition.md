# Composing skaters
View as [web page](https://microprediction.github.io/timemachines/interface). Docs [home](https://github.com/microprediction/timemachines/blob/main/docs/README.md) for abstract description of a skater. 
   
### Residual chaser factory
The [residual_chaser_factory](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/composition/residualcomposition.py) allows you to
use a skater f2 to predict the errors of f1, thus creating a potentially more accurate skater than f1. 

You can use it like this: 

    from timemachines.skaters.proph.prophskaterssingular import fbprophet_univariate as f1
    from timemachines.skaters.simple.hypocratic import quickly_hypocratic as f2 
    from timemachines.skatertools.composition.residualcomposition import residual_chaser_factory
    def f3(y,s,k,a,t,e,r):
        return residual_chaser_factory(y=y, s=s, k=k, a=a, t=t, e=e, f1=f1,f2=f2,r2=r)

Then use f3 just as you would any skater:
    
    import numpy as np 
    y = np.random.randn(100)
    for yi in y:
        xi, x_std, s = f3(y=yi, s=s, k=3)

### Examples

See [skaters/proph/prophetskaterscomposed.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/proph/prophskaterscomposed.py) or 
[skaters/simple/hypocratic](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/hypocratic.py). 

### Article

See [Chasing StatsForecast AutoARIMA Residuals in Two Lines of Code](https://microprediction.medium.com/chasing-statsforecast-autoarima-residuals-in-two-lines-of-code-8a39c8c2561f) for an ARIMA example. 

-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
