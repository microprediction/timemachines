## Skaters
View as [web page](https://microprediction.github.io/timemachines/skaters) or [source](https://github.com/microprediction/timemachines/blob/main/docs/skaters.md).
  
### Finding a skater
Poke around in [/skaters](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters). Here's an incomplete list:

| Package | Location                                                                                            |
|---------|-----------------------------------------------------------------------------------------------------|
| StatsModels.TSA  | [skaters/tsa](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/tsa)     |
| DARTS   | [skaters/drts](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/drts) |
| SkTime  | [skaters/sk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/sk)     |
| StatsForecast  | Also in sktime, for now                                                                      |
| TBATS  | [skaters/bats](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/bats)     |
| PMD  | [skaters/sk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/pmd)     |
| Prophet  | [skaters/proph](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/proph) | 
| NeuralProphet  | [skaters/nproph](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/nproph)     |
| KATS  | [skaters/kts](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/kts)     |
| GreyKite  | [skaters/gk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/gk)     |
| Orbit-ML  | [skaters/orbt](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/orbt)     |
| PyCaret  | [skaters/pycrt](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/pycrt)     |
| River  | [skaters/rvr](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/rvr)     |
| Divinity | [skaters/divine](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/divine) |
| PyDLM | [skaters/pydlm](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/pydlm) |
| PyFlux | [skaters/flux](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/flux) |
| GluonTS | [skaters/glu](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/glu) |
| Merlion | [skaters/mrln](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/mrln) |

And some home grown

| Package | Location                                                                                            |
|---------|-----------------------------------------------------------------------------------------------------|
| Simple  | [skaters/simple](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/simple)     |
| SMDK  | [skaters/smdk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/smdk)     |
| Elo  | [skaters/elo](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/elo)     |

and maybe more (see [here](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters)). 

### Importing a skater
Examples:

    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    from timemachines.skaters.simple.hypocraticensemble import quick_balanced_ema_ensemble as f
    
### Importing a skater by name
Less efficient:

    from timemachines.skaters.localskaters import local_skater_from_name
    f = local_skater_from_name('rvr_p1_d0_q0')
    
Names of skaters appear on the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/faster.html) tables. 
    
### Using a skater 

"Skater" is a nmemonic for the arguments, although you might need only "s" and "k". The script [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py) illustrates the usage pattern. Like so:

    from timemachines.skaters.simple.thinking import thinking_slow_and_fast 
    import numpy as np
    y = np.cumsum(np.random.randn(1000))
    s = {}
    x = list()
    for yi in y:
        xi, x_std, s = thinking_slow_and_fast(y=yi, s=s, k=3)
        x.append(xi)
     
There's more in [examples/basic_usage](https://github.com/microprediction/timemachines/tree/main/examples/basic_usage).
   
### What's a "skater", abstractly?
There's a reason this is called the timemachines package. Skaters are stateless function *suggesting* state machines. 

$$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
$$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. If you prefer an legit (stateful) state machine, see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 


### See also [Skater interface](https://microprediction.github.io/timemachines/interface.html)
for more details about expectations placed on skater functions.

-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
