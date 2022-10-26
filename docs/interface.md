
# Skater interface
View as [web page](https://microprediction.github.io/timemachines/interface) or [source](https://github.com/microprediction/timemachines/blob/main/docs/README.md).
  
This page describes the conventions for skater functions. 
  
### Introduction to skater function conventions

Recall that name *timemachines* is chosen because the skater functions *suggest* state machines for sequential assimilation of observations (as a data point arrives, 
    forecasts for 1,2,...,k steps ahead, with corresponding standard deviations are emitted). However unlike state machines that save state themselves, here the *caller* is expected to maintain state from one 
    invocation (data point) to the next. See the [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) if this seems odd. 

So, that's what's going on with the *s*. As for the k-a-t-e-r:  

      x, w, s = f(   y:Union[float,[float]],             # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:[float]=None,                          # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for hype(r)-pa(r)amete(r)s) 

Since only one *y* arrives at a time, it is up to you to harvest a sequence of the model predictions, if you wish to, as follows:

    def posteriors(f,y):
        s = {}       
        x = list()
        for yi in y: 
            xi, xi_std, s = f(yi,s)
            x.append(xi)
        return x

Though you *could* use the prominently positioned [skating.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skating.py) utilities for processing full histories.  
 
### Skater "y" argument

A skater function *f* takes a vector *y*, where:

    - The quantity to be predicted (target) is y[0] and,
    - There may be other, simultaneously observed variables y[1:] deemed helpful in predicting y[0].

### Skater "s" argument
 
The internal state of the skater, intended to summarize everything the skater needs to know from the past. It is entirely up to the creator of the skater to devise a sensible scheme for s, and it is advised that this be a dictionary that is easily dumped to JSON, although it is impossible to enforce this constraint when incorporating third party packages. 

### Skater "k" argument 

The integer k argument determines the length of the vector of predictions (and also their standard deviations) that will be returned. 

Typically if you really care about k=1 step ahead prediction then you should specify k=1. You *could* specify k=5 or whatever and take the first element of the 5-vector returned, but many skaters might not ensure this is the same as when k=1 is specified (for computational reasons some interpolation might be applied, for instance). 

### Skater "a" argument 

A vector of known-in-advance variables. For instance a day of week. 

You can also use the "a" argument for conditional prediction. This is a matter of interpretation. For instance, you might ask for two predictions, one with a=0 and one with a=1 where this represents rain or something. 

It must be said that at present, most skaters will ignore the "a" argument entirely. 

Also, there are at present no universal conventions for values taken by "a". It is entirely up to the skater to infer this. 

### Skater "t" argument 

Epoch time of the observation. 

Again, many skaters will choose to ignore this input and instead presume that data is regularly sampled. 

### Skater "e" argument ("expiry")

A loose convention but:

       e < 0    -  Tells skater that it should update the state but the actual emitted result won't be used. 
       e = 0    -  Tells skater that the result will matter, so be sure to compute it. 
       e > 0    -  Tells the skater that it has plenty of time, so maybe performing a periodic "fit" is in order, if that's something it does. 
       
This can be very useful for testing, since we can set e<0 during burn-in. 
   
### Skater "r" argument (stands for "hype(r) pa(r)amete(r)s for p(r)e-skaters only)

A real skater doesn't have any hyper-parameters. It's the job of the designer to make it fully autonomous. The small concession made here is the notion of a pre-skater: one with a single float hyperparameter in the closed interval \[0,1\]. Pre-skaters squish all tunable parameters into this interval. That's a bit tricky, so some rudimentary conventions and space-filling functions are provided. See [tuning](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools/tuning) and longer discussion below. 

## Return values
All skater functions return two vectors and the posterior state dictionary. 

      1. The first set of *k* numbers can be *interpreted* as a point estimate (but need not be)
      2. The second is *typically* suggestive of a symmetric error std, or width. However a broader interpretation is possible wherein a skater *suggests* a useful affine transformation of the incoming data and nothing more.  
      3. The third returned value is state, and the skater expects to receive it again on the next call.


          -> x     [float],    # A vector of point estimates, or anchor points, or theos
             x_std [float]     # A vector of "scale" quantities (such as a standard deviation of expected forecast errors) 
             s    Any,         # Posterior state, intended for safe keeping by the callee until the next invocation 
                       
For many skaters the x_std is, as is suggested, indicative of one standard deivation. 

     x, x_std, x = f( .... )   # skater
     x_up = [ xi+xstdi for xi,xstdi in zip(x,xstd) ]
     x_dn = [ xi-xstdi for xi,xstdi in zip(x,xstd) ]
     
then very roughly the k'th next value should, with 5 out of 6 times, below the k'th entry in x_up 
There isn't any capability to indicate three numbers (e.g. for asymmetric conf intervals around the mean).  

In returning state, the intent is that the *caller* might carry the state from one invocation to the next verbatim. This is arguably more convenient than having the predicting object maintain state, because the caller can "freeze" the state as they see fit, as 
when making conditional predictions. This also eyes lambda-based deployments and *encourages* tidy use of internal state - not that we succeed
 when calling down to statsmodels (though prophet, and others including the home grown models use simple dictionaries, making serialization trivial). 
    
### Summary of conventions: 

- State
    - The caller, not the callee, persists state from one invocation to the next
    - The caller passes s={} the first time, and the callee initializes state
    - State can be mutable for efficiency (e.g. it might be a long buffer) or not. 
    - State should, ideally, be JSON-friendly. 
       
- Observations: target, and contemporaneous exogenous
     - If y is a vector, the target is the first element y[0]
     - The elements y[1:] are contemporaneous exogenous variables, *not known in advance*.  
     - Missing data can be supplied to some skaters, as np.nan.  
     - Most skaters will accept scalar *y* and *a* if there is only one of either. 
    
- Variables known k-steps in advance, or conditioning variables:
     - Pass the *vector* argument *a* that will occur in k-steps time (not the contemporaneous one)
     - Remark: In the case of k=1 there are different interpretations that are possible beyond "business day", such as "size of a trade" or "joystick up" etc. 

- Expiry
     - The "e" parameter serves as a performance hack  
     - e<0 tells the skater that the output won't be used, so worry only about state updating
     - e=0 tells the skater that there isn't much time, so this particular invocation isn't the best time to do a periodic "fit" exercise. 

- Hyper-Parameter space (for "pre-skaters")
     - A float *r* in (0,1). 
     - This package provides functions *to_space* and *from_space*, for expanding to R^n using space filling curves, so that the callee's (hyper) parameter optimization can still exploit geometry, if it wants to.   
     - See [tuning](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools/tuning)
     
See [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) or file an issue if anything offends you greatly. 
 
### Aside: more on the e argument ...
Now back to *e* again. 

Some skaters are so fast that a separate notion of 'fit' versus 'update' is irrelevant. Other skaters will periodically fit whether or not e>0 is passed. 
For some, there is even more graduated performance so *e* could be interpreted as "number of seconds allowed". To be safe the tests often pass an e sequence like the following: 

     -1, -1, -1, ... -1 1000 1000 1000 1000 1000 ...
     
because it wants to allow the skaters to receive some history before they are evaluated. Often you just wan the skater to shove that in a buffer and not compute anything, if that is its style, becaue waiting for Facebook prophet to fit itself 500 times is a bit like waiting for the second coming of Christ. 


-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
