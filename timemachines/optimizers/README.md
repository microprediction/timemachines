
# Optimizers

Provides a number of popular derivative free global optimizers with a common, simple calling convention. 

Assumes:

- Function is defined on the unit hyper-cube. 
- Single objective optimization only

Refer to the original packages if you need more advanced functionality. 

### Preliminary study

You may be interested in the [Comparision of Global Optimizers](https://www.microprediction.com/blog/optimize). However this repo
intends to fix some of the issues with that study. 

### Example

To run every optimizer against a number of objective functions

    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in OPTIMIZERS:
            print(optimizer.__name__,(optimizer.__name__,optimizer(objective, n_trials=50, n_dim=5, with_count=True)))

