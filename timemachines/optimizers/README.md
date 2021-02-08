
# Optimizers

Provides a number of popular derivative free global optimizers with a common, simple calling convention. 


### Usage 

    from timemachines.optimizers.pysotcube import pysot_ei_cube
    from timemachiens.optimizers.objectives import schwefel_on_cube
    best_val, best_x, fevals = pysot_ei_cube(schwefel_on_cube, n_trials=16, n_dim=5, with_count=True)

The argument with_count must be set to true if you want three return values, including the function evaluation count, rather than two. 

### Limitations:

- Function is defined on the unit hyper-cube. 
- Single objective optimization only

Refer to the original packages, or create a map from your domain to the cube, if you need more advanced functionality. You can borrow 
from the of the skater domain mappings if you wish. (TODO: )

### Preliminary study

You may be interested in the [Comparision of Global Optimizers](https://www.microprediction.com/blog/optimize). However this repo
intends to fix some of the issues with that study. 

### Elo Ratings

Just run the script elo_ratings.py to initiate an endless sequence of head-to-head optimizer battles, with 
elo ratings updated after each 'game'. 

    from timemachimes.optimizers.eloratings import demo_optimizer_elo
    demo_optimizer_elo()
   
### Run every optimizer against your problem

Or lots of objectives. Modify this pattern as you heed:

    from timemachines.optimizers.alloptimizers import OPTIMIZERS   
    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for optimizer in OPTIMIZERS:
            print(optimizer.__name__,(optimizer.__name__,optimizer(objective, n_trials=50, n_dim=5, with_count=True)))

