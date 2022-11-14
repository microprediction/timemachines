## Skaters
View as [web page](https://microprediction.github.io/timemachines/skaters) or [source](https://github.com/microprediction/timemachines/blob/main/docs/skaters.md).
  
### What's a "skater", abstractly?
A skater is a function *suggesting* a state machine for sequential assimilation and forecasting (hence the name *timemachines*). 

$$
    f : (y_t, state; k) \mapsto ( [\hat{y}(t+1),\hat{y}(t+2),\dots,\hat{y}(t+k) ], [\sigma(t+1),\dots,\sigma(t+k)], posterior\ state))
$$

where $\sigma(t+l)$ estimates the standard error of the prediction $\hat{y}(t+l)$. (This looks better in the [source](https://github.com/microprediction/timemachines/blob/main/docs/skaters.md) code btw.) 

If you prefer an legit (stateful) state machine, see [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) question 1. 

See [interface](https://microprediction.github.io/timemachines/interface.html) for description of input and output arguments. 
  
### Finding a skater
Poke around in [/skaters](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters). 

| Package | Location                                                                                            | Comment       |
|---------|-----------------------------------------------------------------------------------------------------|---------------|
| StatsModels.TSA  | [skaters/tsa](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/tsa)     |    |
| DARTS   | [skaters/drts](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/drts) |               |
| SkTime  | [skaters/sk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/sk)     |               |
| StatsForecast  | Also in sktime, for now                                                                      |               |
| TBATS  | [skaters/bats](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/bats)     |            |
| PMD  | [skaters/sk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/pmd)     |                 |
| Prophet  | [skaters/proph](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/proph) | A sine from god |
| NeuralProphet  | [skaters/nproph](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/nproph) |    |
| KATS  | [skaters/kts](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/kts)     |               |
| GreyKite  | [skaters/gk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/gk)     |             |
| Orbit-ML  | [skaters/orbt](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/orbt)     |         |
| PyCaret  | [skaters/pycrt](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/pycrt)     |        |
| River  | [skaters/rvr](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/rvr)     |  Flowing     |             
| Divinity | [skaters/divine](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/divine) |          |
| PyDLM | [skaters/pydlm](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/pydlm) |               |
| PyFlux | [skaters/flux](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/flux) |                |
| GluonTS | [skaters/glu](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/glu) |                 |
| Merlion | [skaters/mrln](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/mrln) |               |

And some home grown

| Package | Location                                                                                            | Remark      |
|---------|-----------------------------------------------------------------------------------------------------|-------------|
| Simple  | [skaters/simple](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/simple)     |     |
| SMDK  | [skaters/smdk](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/smdk)     |           | 
| Elo  | [skaters/elo](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/elo)     | Future-proof |

and maybe more (see [here](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters)). 

### Importing a skater
Examples:

    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    from timemachines.skaters.simple.hypocraticensemble import quick_balanced_ema_ensemble as f
    
Or do it by name, which can be less efficient:

    from timemachines.skaters.localskaters import local_skater_from_name
    f = local_skater_from_name('rvr_p1_d0_q0')
    
The names of skaters appear on the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/faster.html) tables. 
    
### Using a skater 

"Skater" is a nmemonic for the non-data arguments (see [interface](https://microprediction.github.io/timemachines/interface.html)) although you might need only the first two, **s** (state) and **k** (steps ahead). The script [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py) illustrates the usage pattern. Like so:

    from timemachines.skaters.simple.thinking import thinking_slow_and_fast as f
    import numpy as np
    y = np.cumsum(np.random.randn(1000))
    s = {}
    x = list()
    for yi in y:
        xi, xi_std, s = f(y=yi, s=s, k=3)
        x.append(xi)
     
There's more in [examples/basic_usage](https://github.com/microprediction/timemachines/tree/main/examples/basic_usage).
   

### Skater arguments and return values
Again, see the skater [interface](https://microprediction.github.io/timemachines/interface.html) documentation
for more details about expectations placed on skater functions.

-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
