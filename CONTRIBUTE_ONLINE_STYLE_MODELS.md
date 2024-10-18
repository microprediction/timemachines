
## Contribution pattern:  new batch-style time-series package

Do you have a favourite Python package for time-series analysis that you would like to see included in the funnel?

### Likely contributor

- You maintain a time-series package, or just really like one. 
- The package exposes a way of predicting time-series in an incremental fashion, meaning that you don't have to start all over again every time the length of the time-series increases by one. 

### When not to use this pattern

If the package assumes you have to fit every new data point (like Prophet or many others) then see [CONTRIBUTE_BATCH_STYLE_MODELS.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_BATCH_STYLE_MODELS.md) instead as this will provide a much shorter path.  

### How to contribute 

1. (Optional) Join crunch discord (invite [here](https://github.com/microprediction/monteprediction/blob/main/TFRO.md))
2. Grok the package you think should be in. Create an example colab notebook (like [examples here](https://github.com/microprediction/timeseries-notebooks)) that uses the package. It should show how to produce a k-vector of 1..k step ahead predictions. You'd be surprised at how many packages seem to think this is an obscure use case and don't include it in their README :)

At this point you've already helped a lot. If you want to take it all the way...

 3. Choose a short PREFIX that isn't exactly the same as the library (here PREFIX='sk', obviously)
 4. Write PREFIXinclusion.py           (in the same style as [skinclusion.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skinclusion.py))
 5. Write PREFIXwrappers.py            (expose the batch functionality in the same style as [skwrappers.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skwrappers.py))

Thus far this has exposed the 3rd party library in an offline manner (i.e. expects fit to be called each data point). Now you need to turn this into a skater. 

 6. Read or re-read the main [README.md](https://github.com/microprediction/timemachines) that explains the skater signature, and see also the notes below that relate directly to online style skaters. 
 7. Read the [movingaverage.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/movingaverage.py) example and, in particular, note the use of the parade helper. 
 8. For the convenience of others, adopt the style of putting a mini-test at the bottom, as with [movingaverage.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/movingaverage.py)
 
Next you need to register a list of fully autonomous skaters
 
 9. Create a file similar to [allriverskaters](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/rvr/allriverskaters.py) that collects all the skaters you have created.      
 
Write unit tests

 10. Cut and paste the example unit test [test_sk_random_skater](https://github.com/microprediction/timemachines/blob/main/tests/sk/test_sk_random_skater.py)

Run pytest and the individual test repeatedly. When it is working well you can include it in the master list:
 
 11. Modify [localskaters.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/localskaters.py) 
 12. Modify [README.md](https://github.com/microprediction/timemachines/blob/main/README.md) to include a suggestion to users to pip install the new requirements
 13. Modify [setup.py](https://github.com/microprediction/timemachines/blob/main/setup.py) to include the new directory.  


## Understanding online style skaters
The directory [simple](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/simple) is intended to contain some examples of creating online style skaters. Here is a quick review of some items from the [README](https://github.com/microprediction/timemachines) that might help understand what a skater is and what it is not. The good news is that a skater is merely a function. 
   
             x, w, s = f(   y:Union[float,[float]],       # Contemporaneously observered data, 
                                                      # ... including exogenous variables in y[1:], if any. 
                  s=None,                                  # Prior state
                  k:float=1,                               # Number of steps ahead to forecast. Typically integer. 
                  a:[float]=None,                          # Variable(s) known in advance, or conditioning
                  t:float=None,                            # Time of observation (epoch seconds)
                  e:float=None,                            # Non-binding maximal computation time ("e for expiry"), in seconds
                  r:float=None)

   - Your function must take either a scalar y or a list. If the latter, your skater should interpret the first entry y[0] as the quantity
   that needs forecasting, whereas y[1:] are merely helpful. If your model doesn't consider exogenous variables then the first line should probably be:
   
   
           y0 = wrap(y)[0]    
   
   
   - Your function must return a list or vector x of length k where x[0] is 1 step ahead, x[1] is 2 steps ahead and so forth. Ideally this is done in 
   a fast, incremental manner. Every time a number arrives the predictions for the next k are spat out. It is okay to create skaters that are slow and
   use packages that are designed for more one-off tabular use - since it is helpful to be able to benchmark fast skaters against slow ones.

   - Your function must also return a second list w that will be interpreted (loosely) as a 1-standard deviation error in the skater's forecast. It 
   is not absolutely necessary to fret about this. Some skaters just return [1 1 1 ... 1]. However, it is just a couple of lines of code to include
   a skater's own empirical estimate of its own accuracy and this is extremely important to do if you want your skater to be included in 
   ensembles, stacks and so forth in a productive manner. For this reason you are strongly advised to study the pattern adopted in 
   [movingaverage.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/movingaverage.py) and in particular
   the use of the *parade*:
   
       
          x = [y0]*k            # What a great prediction !
          bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)  # update residual queue
          x_std_fallback = nonecast(x_std, fill_value=1.0)
          return x, x_std_fallback, s
      
      
 Aside: You can read the code for parade in [/components/parade.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/components/parade.py) and, as stated there,
    the parade manages k-step ahead forecasts and their comparison to actual data as it arrives. It is called a parade because it is a procession of predictions
     that are waiting to be judged when data arrives. The predictions are stored in an array where the n'th entry will be judged after n data points arrive. There
     are multiple entries in each row of the parade. For example if data is sampled on the hour, then a 3-hour ahead forecast made at 11:01 am will march alongside
      a 5-hour ahead prediction made two hours earlier at 9:00am. Both will be judged at 2pm. However they are tagged with the prediction horizon, allowing
    for rolling statistics to be tracked separately for 1-hr, 2-hr,...10-hr ahead predictions, say. Anyway ... YOU DON'T NEED TO CARE ABOUT THIS but you do probably need
    to include the line 
    
    
           bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0) 
    
    
  in your skater unless you have a better way to estimate the skater's error. End of aside.  

   - Your skater must also return whatever state *s* it will need on the next invocation. That's entirely up to you - although notice that in the example above the entry
   s['p'] is being used to store the prediction parade. The state might include a buffer of recent observations, or some moving average, or what-have-you. The only real convention here is
   that *ideally* the state should not puke when you call json.dumps(s). Where possible use simple dictionaries, lists etc. You will also have observed from your careful study
   of the [README](https://github.com/microprediction/timemachines) that when the caller to your function passes an empty dictionary as state, that's your cue to instantiate whatever needs
   instantiation on the first call. Perhaps a clean example of this initialization is [tsaconstant.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/tsa/tsaconstant.py) where
   it is, I hope, pretty obvious that this block of code warms up the state s:
   
   
           if not s.get('y'):
           s = {'y': list(),
               'a': list(),
               'k': k,
               'p':{}}

 
   Actually it doesn't do much, as you can also see, because it doesn't want to spend energy estimating a time-series model until there is enough data. Which brings me to the e argument
   
   - Your skater should interpret the e argument before embarking on expensive computation. 
   
      * e < 0     ... skater should not do anything energetic <-- treat this as a burn-in period
      * e = 0     ... skater should do what it pleases (though presumably respond reasonably quickly!)
      * e = 55.7  ... skater should feel free to spend 55.7 seconds performing some periodic refitting, or whatever. 
      * e =100    ... skater should feel free to spend 100 seconds performing some periodic refitting, or whatever. 

   Bear in mind that there is no convention for providing a skater with historical data beyond this. The only way for the caller to do this is by passing
   the historical data one point at a time with e<0 set. Obviously, the skater code doesn't know what machine it will be run on but that's too bad. 
   
  - Your skater might also try to interpret known-in advance variables. These are things like day-of-week that are known k-steps in advance. The thing to remember here,
  that is emphasized in the README but I will emphasize again, is that the caller will pass numbers in the *a* argument (a scalar or vector) that pertain to the future time 
  of the longest forecast (i.e. k steps ahead) and *not the contemporaneous moment*. So in this way *a* is very different from y[1:] and the two should not be confused. If you
  want an example of a skater using known-in-advance variables then perhaps see [prophet skater](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/proph)
  which wraps Facebook's Prophet package.
  
   
  That's what you need to know. I'll summarize with a C&P from the README:  

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

  - Hyper-Parameter space. Only relevant if you wish to create "pre-skaters" which are, loosely, "almost-skaters". 
     - A float *r* in (0,1). 
     - This package provides functions *to_space* and *from_space*, for expanding to R^n using space filling curves, so that the callee's (hyper) parameter optimization can still exploit geometry, if it wants to.   
     
See [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) or file an issue if anything offends you greatly. 
 
## On testing a skater function 

At minimum, please add a test script like [tests/simple/test_simple.py](https://github.com/microprediction/timemachines/blob/main/tests/simple/test_simple.py). This one uses a canned time-series of hospital wait times. 

    from timemachines.skaters.simple.movingaverage import precision_ema_ensemble, aggressive_ema_ensemble
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error

    SIMPLE_TO_TEST = [ precision_ema_ensemble, aggressive_ema_ensemble ]

    def test_ensemble():
        for f in SIMPLE_TO_TEST:
           err = hospital_mean_square_error(f=f, k=5, n=150)

Please ensure this test runs quickly. 

## More on hyper-optimizing a skater function 

This is up to you. If you need real-world data, you have it. Peruse the [knowledge center](https://www.microprediction.com/knowledge-center) or jump to this [notebook](https://github.com/microprediction/microprediction/blob/master/notebook_examples/Python_Module_3_Getting_History.ipynb) showing
how to retrieve recently created live data. As noted, you're advised to use the fresh bait to avoid over-fitting and sneaky kinds of self-deception that arise 
over time when historical data is used.  

You may have noticed the *r* argument in the skater signature. Your skater (or skater factory) *might* want to accept an *r* value if you think there is value in hyper-parameters. However the intent of the time-machines library is the provision of 
  a number of purely autonomous forecasting methods, so if *r* is extremely important and needs to be fit for each and every time-series your method might be a little out of scope. That's
  not going to be hugely useful to other automated uses of your skater, generally speaking, and there might be no human around to interpret what you meant. So a pure skater, or a
  "bound skater" as we might call them has no *r* parameter. 
  
  All that said, there is just a tiny amount of wiggle room permitted and you might want to create a family of
  skaters parametrized by a SINGLE SCALAR *r* in the interval [0,1]. In addition, you are welcome to use the space-filling curve conventions and mappings provided in this package to wiggle yourself
  to more freedom - say a cube for hyper-parameters instead of the interval. Either way there is a decent chance that some black-box algorithm can be expected to optimize your
  hyper-parameters if run over many real-world time-series. One way to do this is by using the [humpday](https://github.com/microprediction/humpday) package, or merely cutting and pasting
  the examples therein for convenient use of popular and not-so-popular search algorithms. See the [article](https://www.microprediction.com/blog/humpday) about humpday or 
  just the HumpDay [readme](https://github.com/microprediction/humpday). 


## Steps to incorporate skater requirements into the package setup and requirements.txt 

Don't. 

I'm moving away from including 3rd party packages in the setup requirements. If your skater needs a package named infrequently_maintained_and_certain_to_crash_on_m1 I would rather you create a file like
[pmdinclusion.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/pmd/pmdinclusion.py) and then modify the section of the readme, advising folks that they
can pip install the package should they wish to be able to use your skater. 

## Steps to incorporate a skater into the Elo ratings

This part should get you excited, and it is pretty automatic. Of course, you want to see your skater climb the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) which are a by-product of regular testing of the package. I use Elo ratings because they are incremental and we can avoid putting strain on the servers provided by our friends at GitHub who support
open source work like this. 

You don't need to do much for this, merely include a list of 'bound' skaters (i.e. no hyper-param *r* expected) in a fashion similar to what I've done for 
[allpmdskaters](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/dlm/alldlmskaters.py). Finally, you can reference this when making a change to 
[skaters/allskaters.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/allskaters.py) which is the list of all skater lists. See the previous comment about not assuming
existence of a package, however. Your list should be empty if your skater uses the package infrequently_maintained_and_certain_to_crash_on_m1 and the user has not pip installed that. You should also edit [setup.py](https://github.com/microprediction/timemachines/blob/main/setup.py) to include your director, 


## Steps to use your skater to win cash at microprediction.com 

Obviously the dismal amounts of prizemoney we offer pale against the immense satisfaction you will receive from advancing mankind's collective time series prediction capability, but you need look no further than the leaderboard
at microprediction for some example of crawlers that use the timemachines package. Unfortunately there's more to distributional prediction than point estimates, but nonetheless you are welcome to 
set a crawler loose that is informed in some way. You can see the pattern in the crawler called [Datable LLama](https://github.com/microprediction/microprediction/blob/master/crawler_examples/datable_llama.py). 

    from microprediction.config_private import DATABLE_LLAMA
    from microprediction.streamskater import StreamSkater
    from timemachines.skaters.simple.movingaverage import precision_ema_ensemble
    
    if __name__=='__main__':
        skater = StreamSkater(write_key=DATABLE_LLAMA, f=precision_ema_ensemble, use_std=True, max_active=1000)
        skater.set_repository(
            'https://github.com/microprediction/microprediction/blob/master/crawler_examples/datable_llama.py')
        skater.run()
        
    

As you can see this leans heavily on the StreamSkater class. This an other methods are explained in the [crawler examples](https://github.com/microprediction/microprediction/tree/master/crawler_examples) folder. 

## More help / discussion

As noted, I try to jump on a Google Meet twice a week and the details are in the microprediction [knowledge center](https://www.microprediction.com/knowledge-center). My arrival rate is higher on Fridays than Tuesdays :)

I'm not so good at scheduling calls outside of these times and frankly that tends to be counter to my productivity anyway. So just jump on some Tuesday night or 
Friday noon if you are keen to contribute to this package, or anything else that relates to open source community prediction. 


![](https://i.imgur.com/l14hKmr.jpg)
