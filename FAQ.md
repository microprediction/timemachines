

### FAQ 0: This is great, how can I help?
    
See  [CONTRIBUTE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE.md)
    
    
### FAQ 1: Why not have the model persist the state?

Answer (A): Well, you can trivially turn any skater function into a callable that does that, should you wish: 

       class Predictor:
   
           def __init__(self,f):
                self.f = f
                self.s = s

           def __call__(self,y,k,a,t,e):
                x, x_std, self.s = self.f(y=y,s=self.s,k=k,a=a,t=t,e=e)
                return x, x_std

or write a decorator. Whatever, it's Python. 

Answer (B): The intent is to produce simple web-service friendly models. 

Answer (C): It's "pure". Sort of. Dict are mutable. Let's say it is as pure as the driven snow that has turned in to sludge in New York City. 

Answer (D): The caller can request conditional predictions easily (if they remember to do a deep copy of the state - see this [note](https://github.com/microprediction/timemachines/issues/51) ). 

### FAQ 2: What's with the z-curves and *r* parameter?

The idea is that *any* black-box optimizer should be able to optimize *any* time-series "pre-skater" without a human providing interpretation of hyper-parameters. So we force pre-skaters to expose at most a single scalar hyper-parmeter, but also supply suggested space-filling curves. This...

- is hopefully a *reasonable* way to map the most important hyper-parameter choices,
- that imposes some geometric discipline on the hyper-parameter space in the first place, and

and, as noted, facilitates comparison of different ways to search hyper-parameters, across packages which have *entirely different conventions* and hyper-parameter spaces. 

### FAQ 3: Why not use the "big name" packages, like prophet, directly?

Answer (A) They kinda suck, for one thing. Perhaps that's why 15,000 people read [this article](https://www.microprediction.com/blog/prophet) in a single day. Don't be the last.  

Answer (B) In all seriousness, knock yourself out. Use whatever you like. Observe that this package wraps *some* functionality, not all by any means, of some popular open source packages. But honestly the slow ones are here, along with the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html), mostly to reassure you that speed comes for free. You're probably gaining accuracy, not losing it, when you adopt a faster algorithm and ignore the latest exhortation to use FAANG-XYZ, written by some noob on Towards Data Science.   

Answer (C). Given the above, you might not want to wed yourself to the quirks of a particular packages' API. You *might* like this package if you want to be able to do this:

        s,k = {}, 3
        for yi,ai in zip(y,a[k:]): 
            xi, xi_std, s = f(y=yi,s=s,k=k,a=ai)

and swap out *f* when it doesn't perform for you. 

Notice what isn't here: 
 - Pandas dataframes
 - A long list of methods and properties 
 - Column naming conventions 
 - The customary 10-50 lines of setup code before a prediction can be made,
 - The customary need to trace into the code to infer intent, including which parameters are supposed to be supplied. 
 - Possible confusion between variables known in advance and those observed contemporaneously,
 - Possible confusion about prediction horizon,
 - Possible conflation of 3-step ahead prediction with the 1-step ahead prediction applied three times, 
 - Datetime manipulation, and conventions like '5min' which not everyone agrees on. 


### FAQ 4: What are the drawbacks? 
Autonomous prediction is hard and there's only subsidized lunches. 

There are also limitations of the skater approach. The simple data model in *y*, *a* is not well suited to problems where exogenous data comes and goes, and therefore cannot easily be represented by a vector of fixed length (you might consider a dictionary interface instead, as with
the river package). 

There is, at time of writing, also a shortage of skaters that exploit anything beyond the first entry in 'y' (exogenous vars), or use 'a' (known in advance). So think of this package as mostly univariate, for now. 

### FAQ 5: Why do skaters only return two numbers per horizon, instead of a full distribution? 

Yes, the skater does not return a full distribution - unless you smuggle it into the state. 
However this package was motivated by the desire to create better free turnkey distributional forecasts, at [microprediction.org](https://www.microprediction.org), and you might infer that skaters returning two numbers per horizon might be useful 
as part of a chain of computations that eventually produces a distributional estimate. Skaters can be considered linear transforms
of incoming data, and part of the agenda here is figuring out how to judge skaters
in a manner that better reflects downstream use in distributional estimates. Here the theory of proper scoring rules doesn't really suffice, it would seem. End of aside.  
 
