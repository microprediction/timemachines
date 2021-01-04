# timemachines
Time series prediction models:

- taking the form of *pure functions* (that's the intent, anyway, but its a free country)
- that represent *state machines*
- that are ideal for *lambda deployment*
- in *online settings* 
- where *urgency* may or may not be important and
- *benchmarking* is considered crucial, and made easy because the interface is
- largely *unopinionated* 

Some of these models are used as intermediate steps in the creation of distributional forecasts, at [microprediction.org](www.microprediction.org). 
### The "skate" interface
A "model" is merely a function *suggesting* a state machine, whose role is sequential processing of data and emmission of "something".   

    x, s = f(                                                    # Returns a prediction (or latent var) and posterior state
                        y:Union[float,[float]],                  # Contemporaneously observerd data, 
                                                                 # ... including exogenous variables in y[1:], if any. 
                        s=None,                                  # State
                        k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                        a:Union(float,[float])=None,             # Variables known in advance
                        t:float=None,                            # Time of observation (epoch seconds)
                        e:float=None) -> Union(float,[float])    # Non-binding maximal computation time ("e for expiry"), in seconds
    
To emphasize, every model in this collection is *just* a function and the intent is that these functions are pure. 

### Usage example
Given a "model" f, we can process observations xs as follows:

    def posteriors(f,ys:[float]):
        s = None
        xs = list()
        for t,y in data.items()
            x, s = f(y,s)
            xs.append(xs)
        return xs
    
### Conventions 

- The format taken by state is determined by the model, not caller
- The caller passes s=None the first time
       
- If returning a single value:
     - This should be an estimate of y[0] if y is a vector. 
     - The elements y[1:] are to be treated as exogenous variables, not known in advance. 

- The scalar, or vector a is used to pass "known in advance" variables
     
- Missing data passed as np.nan 
      
- State can be mutable for efficiency (e.g. it might be a long buffer) or not. Recall that Python is pass-by-object-reference. 
      - Caller should not need to know anything about state
      - Reponsibility and ownership lies with the function
      - State is not an invitation to sneak in additional arguments
      
- If y=None is passed, it is a suggestion to the "model" that it has time to perform some
      offline task, such as a periodic fitting. 
      - In this case it would be typical to supply a larger e than usual.
      - The function should return x=None, as acknowledgement that it has recognized the "offline" convention
   
### There is no fit()
See the last convention listed above. 


### There are no classes in this package at all
But you can make them. For example:

       class Predictor:
   
           def __init__(self,f):
                self.f = f
                self.s = s

           def __call__(self,y,k,a,t,e):
                x, self.s = self.f(y=y,s=self.s,k=k,a=a,t=t,e=e)
                return x
            

### What's not in the interface
This package wraps some time series prediction libraries that:

 - Use pandas dataframes
 - Bundle data with prediction logic
 - Rely on column naming conventions 
 - Require 10-20 lines of setup code before a prediction can be made
 - Require tracing into the code to infer intent
 - Require conventions such as '5min' 

Just observing, not judging. Depending on your task you may prefer the underlying libraries and additional functionality they bring. 

### Out of scope
The simple interface is not well suited to problems where exogenous data comes and goes. You might consider a dictionary interface instead, as with the river package. It is also not well suited to fixed horizon forecasting if the data isn't sampled terribly regularly. Nor is it well suited to prediction of multiple time series whose sampling occurs irregularly. Ordinal values can be kludged okay, but purely categorical not so much. 

### Yes we're keen to receive PR's
If you'd like to contribute to this standardizing and benchmarking effort, 

- See the [list of popular time series packages](https://www.microprediction.com/blog/popular-timeseries-packages) ranked by download popularity. 
- Or add your own

Consider warming up some of the packages by choosing hyper-parameters and testing on real data. See the ([tutorial](https://www.microprediction.com/python-3)) on retrieving historical data that never gets stale, or the [real data](https://pypi.org/project/realdata/) package. We'd also refer you to the [comparison of hyper-parameter optimization packages](https://www.microprediction.com/blog/optimize). 


