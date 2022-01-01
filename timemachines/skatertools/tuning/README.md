
A skater is morally a "bound" model (i.e. fixed hyper-parameters) and ready to use. Any fitting, estimation or updating is the skater's internal responsibility. 

That said, it is sometimes useful to enlarge the skater concept to include hyper-parameters, as this enourages a more standardized way to expose and fit them. These are called pre-skaters and they admit just one additional argument - the scalar "r" parameter. It remains the responsibility of the pre-skater designer to ensure that the parameter space is folded into (0,1) is a somewhat sensible way. 

The use of a single scalar for hyper-parameters may seem unnatural, but is slighly less unnatural if [conventions](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/utilities/conventions.py) are followed that inflate \[0,1\] into the square \[0,1\]^2 or the cube \[0,1\]^3. See the functions **to_space** and **from_space**. This also makes it easy for anyone to design new black box optimization routines that can work on any skater, without knowing its working. 

The ancilliary package called [HumpDay](https://github.com/microprediction/humpday) provides a comparision of derivative free optimizers that might be useful for fitting pre-skaters, and thereby creating fully autonomous skaters.  

### Provided hyper-parameter optimization utilities 

- Use data from [Microprediction](https://github.com/microprediction/microprediction)
- Example: [optimizing prophet](https://github.com/microprediction/timemachines/blob/main/examples/tuning/optimizing_prophet_live.py)

### Provided space-filling curve conventions

- The default map *from_space* from (0,1)^3 or (0,1)^2->(0,1) uses interleaving of digits in the binary representations (after first scaling).
- The script [demo_param_ordering.py](https://github.com/microprediction/timemachines/blob/master/examples/tuning/demo_param_ordering.py) illustrates
the mapping from r in (0,1) to R^n demonstrating that the first coordinate will vary
more smoothly as we vary r than the second, and so on.  
- If you need dim>3 for hyper-parameters, you can always use *to_space* or *from_space* more than once. 
- There are some functions provided to help you squish your hyper-params into the hypercube. The script [demo_balanced_log_scale.py](https://github.com/microprediction/timemachines/blob/master/examples/hyper/demo_balanced_log_scale.py) illustrates a
quasi-logarithmic parameter mapping from r in (0,1) to R which you can take or leave. 

[![IMAGE ALT TEXT](https://i.imgur.com/4F1oHXR.png)](https://vimeo.com/497113737 "Parameter importance")
Click to see video


### Alternatives

Of course, if you want to optimize skaters by some other means
and want a flexible way to represents searches then use one of the optim packages directly (e.g. optuna is pretty flexible in this regard). 
