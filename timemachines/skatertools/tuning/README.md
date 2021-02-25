
### Hyper-parameter optimization utilities 

- Uses optimizers from [HumpDay](https://github.com/microprediction/humpday)
- Uses data from [Microprediction](https://github.com/microprediction/microprediction)
- Example: [optimizing prophet](https://github.com/microprediction/timemachines/blob/main/examples/tuning/optimizing_prophet_live.py)

### Even more about hyper-parameters

The restriction that all hyper-parameters be represented as r in (0,1) seems harsh. To be slightly less harsh, we include some standard ways
to use (0,1)^2 or (0,1)^3 should that be preferable. Admittedly, this may still not be the most natural way to represent choices, but here
we are trying to give lots of different optimizers a run at the problem. Of course, if you want to optimize skaters by some other means
and want a flexible way to represents searches then use one of the optim packages directly (e.g. optuna is pretty flexible in this regard). On 
the other hand, if you want to benefit from the operational simplicity of r in (0,1) then...

- The default map *from_space* from (0,1)^3 or (0,1)^2->(0,1) uses interleaving of digits in the binary representations (after first scaling).
- The script [demo_param_ordering.py](https://github.com/microprediction/timemachines/blob/master/examples/tuning/demo_param_ordering.py) illustrates
the mapping from r in (0,1) to R^n demonstrating that the first coordinate will vary
more smoothly as we vary r than the second, and so on.  
- If you need dim>3 for hyper-parameters, you can always use *to_space* or *from_space* more than once. 
- There are some functions provided to help you squish your hyper-params into the hypercube. The script [demo_balanced_log_scale.py](https://github.com/microprediction/timemachines/blob/master/examples/hyper/demo_balanced_log_scale.py) illustrates a
quasi-logarithmic parameter mapping from r in (0,1) to R which you can take or leave. 

[![IMAGE ALT TEXT](https://i.imgur.com/4F1oHXR.png)](https://vimeo.com/497113737 "Parameter importance")
Click to see video