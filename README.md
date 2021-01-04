# timemachines
Time series prediction models taking the form of state machines, where the caller is expected to maintain the state. Ideal for lambda deployment. 

### Motivation
This repo attempts to standardize a variety of disparate approaches to time series prediction around a *very* simple functional interface

    y_hat, S' = predict(y:Union[float,[float]],
                        S=None, 
                        k:int=1,
                        t:int=None,
                        a:[float]=None,
                        tau:int=None) -> Union(float,[float])  
    
To emphasize, every model in this collection is *just* a function and the intent is that these functions are pure. 

### Intended usage
Notice that the caller maintains state, not the "model".

    state = None
    predictions = list()
    for t,y in data.items()
        y_hat, state = predict(y,state)
        predictions.append(y_hat)
    
### Requirements on predict
Conventions:

    - Expect S=None the first time, and initialize state as required
    - Tau, with units in seconds, is a suggested maximum computation time
    - The observation time t is epoch seconds. 
    - If returning a single value:
          - This should be an estimate of y[0] if y is a vector. 
          - The elements y[1:] are to be treated as exogenous variables, not known in advance. 
          - The vector a is used to pass "known in advance" variables
    - Missing data passed as np.nan
    - If y=None is passed, it is a suggestion to the "model" that it has time to perform some
      offline task, such as a periodic fitting. In this case it would be typical to supply a
      larger tau than usual. 
   

### What's not here
This wraps some time series prediction libraries that:

     - Use pandas dataframes
     - Bundle data with prediction logic
     - Rely on column naming conventions 
     - 

### Out of scope
The simple interface is not well suited to problems where exogenous data comes and goes. You might consider a dictionary interface instead, as with the river package. It is also not well suited to fixed horizon forecasting if the data isn't sampled terribly regularly. 

