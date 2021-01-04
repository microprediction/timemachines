# timemachines
Time series prediction models taking the form of state machines, where the caller is expected to maintain the state. Ideal for lambda deployment. 

### Motivation
This repo attempts to standardize a variety of disparate approaches to time series prediction around a *very* simple functional interface

    y_hat, S' = predict(y:Union[float,[float]],
                        S=None, 
                        k:int=1,
                        t:int=None,
                        a:[float]=None) -> Union(float,[float])  
    
To emphasize, every model in this collection is *just* a function and the intent is that these functions are pure. 

### Intended usage

    state = None
    predictions = list()
    for t,y in data.items()
        y_hat, state = predict(y,state)
        predictions.append(y_hat)
    
### Requirements on predict
The function (or callable) predict should behave as follows:

    - Expect S=None the first time, and initialize state as required
    - The observation time t is epoch seconds. 
    - If returning a single value:
          - This should be an estimate of y[0] if y is a vector. 
          - The elements y[1:] are to be treated as exogenous variables, not known in advance. 
          - The vector a is used to pass "known in advance" variables
   

### What's not here
This wraps some time series prediction libraries that:

     - Use pandas dataframes
     - Bundle data with prediction logic
     - Rely on column naming conventions 
     - 

### Limitations
The simple interface is not well suited to problems where exogenous data comes and goes. You might consider a dictionary interface instead, as with the river package. 

