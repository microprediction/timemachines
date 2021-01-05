# timemachines
Time series prediction models:

- taking the form of *pure functions* 
- that are recipes for *state machines*
- and ideal for *lambda deployment*
- in *online settings* 
- where helpers maintain state *on their behalf*, and
- where *urgency* may or may not be important and,
- *benchmarking* is considered crucial - not to mention easy 
- because the interface is *mostly unopinionated* 

Some of these models are used as intermediate steps in the creation of distributional forecasts, at [microprediction.org](www.microprediction.org). And yes, this is Python so we can't *enforce* purity. There may be cases where a callable is the right way to do something, but bear in mind these are intended for stateless deployment using "helper" processes that catch and receive state. 

### The "skater" interface
A "model" is merely a function *suggesting* a state machine, whose role is sequential processing of data and emmission of "something" - usually a k-step ahead point estimate or estimate of a latent variable.    

    x, s = f(   y:Union[float,[float]],                  # Contemporaneously observerd data, 
                                                         # ... including exogenous variables in y[1:], if any. 
                s=None,                                  # Prior state
                k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                a:float=None,                            # Variable(s) known in advance, or conditioning
                t:float=None,                            # Time of observation (epoch seconds)
                e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                r:float=None)                            # Hyper-parameters ("r" stands for for R^n)
                      
    
To emphasize, every model in this collection is *just* a function and the intent is that these functions are pure. 

### Usage example
Given a "model" f, also referred to as the callee, we can process observations xs as follows:

    def posteriors(f,ys:[float]):
        s = None
        xs = list()
        for t,y in data.items()
            x, s = f(y,s)
            xs.append(xs)
        return xs
    
### Conventions 

- The caller persists state from one invocation to the next
    - The format taken by state is determined by the callee, not caller
    - The caller passes s=None the first time
    - The function initializes state as necessary, and passes it back
    - The caller keeps the state and sends it back to the callee
    - State can be mutable for efficiency (e.g. it might be a long buffer) or not. 
    - Recall that Python is pass-by-object-reference. 
    - State should, ideally, be JSON-friendly. Use .tolist() on arrays.
    - State is not an invitation to sneak in additional arguments.
       
- Univariate or multivariate observation argument
     - If y is a vector, the target is the first element y[0]
     - The elements y[1:] are contemporaneous exogenous variables, *not known in advance*.  
     - Missing data as np.nan but *not None* (see fitting below)

- Fitting:  
     - If y=None is passed, it is a suggestion to the callee to perform fitting, should that be necessary. 
     - Or some other offline periodic task. 
     - In this case the *e* argument takes on a slightly different interpretation, and should probably
     be considerably larger than usual. 
     - The callee should return x=None, as acknowledgement that it has recognized the "offline" convention
   

- Variables known in advance, or conditioning variables:
     - Passed as *scalar* argument *a* in (0,1). 
     - See discussion below re: space-filling curves so you know this isn't really a huge restriction.  
     - Rationale: make it easier to design general purpose conditional prediction algorithms
     - Bear in mind many functions will ignore this argument, so we have little to lose here. 
     - Caller can deepcopy the state to effect multiple conditional predictions.
     - Example: business day indicator
     - Example: size of a trade
     - Example: joystick button up 

- Parameter space
     - Caller has a limited ability to suggest variation in parameters (or maybe hyper-parameters, since 
     many callees will fit parameters on the fly or when there is time).
     - All parameters must be squished into a single float *r* in (0,1). 
     - Arguably, this makes callees more canonical and, 
     - seriously, there are lots of real numbers, and 
     - the intent here is that the caller shouldn't need to know a lot about parameters
     - This package provides some conventions for expanding to R^n using space filling curves,
     - so that the (hyper) parameter optimization can still exploit geometry, as you see fit. 
      
- Ordering of parameters in space-filling curve
    - The most important variables should be listed first, as they vary more slowly. 
    - See picture below or video
    
### Space-filling conventions for *a* and *r*


The script [demo_balanced_log_scale.py](https://github.com/microprediction/timemachines/blob/master/examples/demo_balanced_log_scale.py) illustrates the
quasi-logarithmic parameter mapping from r in (0,1) to R. 

![](https://i.imgur.com/NCFCTeK.png)


The script [demo_param_ordering.py](https://github.com/microprediction/timemachines/blob/master/examples/demo_param_ordering.py) illustrates
the mapping from r in (0,1) to R^n. Observe why the most important parameter should be listed first. It will vary
more smoothly as we vary r. 

[![IMAGE ALT TEXT](https://i.imgur.com/4F1oHXR.png)](https://vimeo.com/497113737 "Parameter importance")
Click to see video
 
    
### FAQ:

Question 1. Why not have the model persist the state?

Answer 1. Go ahead:

       class Predictor:
   
           def __init__(self,f):
                self.f = f
                self.s = s

           def __call__(self,y,k,a,t,e):
                x, self.s = self.f(y=y,s=self.s,k=k,a=a,t=t,e=e)
                return x

or write a decorator. However we have lambda applications in mind. 

Question 2. Why do it this bare-bones manner with squished parameter spaces?  

Answer 2. The intent is comparison, combination and search for models. However the
hope here is that there is a *reasonable* way to map the most important hyper-parameter choices and thereby search across packages which
have entirely different conventions. This package wraps some time series prediction libraries that:

 - Use pandas dataframes
 - Bundle data with prediction logic
 - Rely on column naming conventions 
 - Require 10-20 lines of setup code before a prediction can be made
 - Require tracing into the code to infer intent
 - Require conventions such as '5min' 

This package should *not* be viewed as an attempt to wrap most of the functionality of these packages. If you 
have patterns in mind that match them, and you are confident of their performance, you are best served to 
use them directly. 

### Scope and limitations
The simple interface is not well suited to problems where exogenous data comes and goes. 
You might consider a dictionary interface instead, as with the river package. 
It is also not well suited to fixed horizon forecasting if the data isn't sampled terribly regularly. 
Nor is it well suited to prediction of multiple time series whose sampling occurs irregularly. 
Ordinal values can be kludged into the parameter space and action argument, but purely categorical not so much. And finally, if you
don't like the idea of hyper-parameters lying in R^n or don't see any obvious embedding, this might 
not be for you. 

### Yes, we're keen to receive PR's
If you'd like to contribute to this standardizing and benchmarking effort, 

- See the [list of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) ranked by download popularity. 
- Or add your own
- Think about the most important hyper-parameters
- Consider "warming up" the mapping (0,1)->hyper-params by testing on real data. 
- See the ([tutorial](https://www.microprediction.com/python-3)) on retrieving historical data that never gets stale.
- See the [real data](https://pypi.org/project/realdata/) package, if that's simpler.
- Perhaps of interest, the [comparison of hyper-parameter optimization packages](https://www.microprediction.com/blog/optimize). If you are the maintainer of a time series package, we'd love your feedback and if you take the time to submit a PR here, do yourself a favor and also enable "supporting" on your repo. 


