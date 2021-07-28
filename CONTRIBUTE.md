
# Time-machines contributors guide

Here are some new notes for contributors to the timemachines package. You can also talk to a human. I try to jump on 
a Google Meet twice a week and the details are in the microprediction [knowledge center](https://www.microprediction.com/knowledge-center). 

Contribution qualifies for monthly [contributor prizes](https://www.microprediction.com/competitions/best-project-contributor) and you can ignore the end-date there if there is one.   

There are two principle ways you can contribute to the timemachines package:

  - Contribute a "skater":
    - A univariate k-step ahead forecaster, preferably one that is fast.
    - For example one that uses a package not already incorporated, possibly from [this list](https://www.microprediction.com/blog/popular-timeseries-packages)
    - Or a clever ensembling or stacking of existing skaters
    
  - Publish live data on an ongoing basis
    - This can help make the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) better.
    - However only "live" data is ideal, to prevent algorithms from memorizing
    - See [instructions for publishing data](https://www.microprediction.com/get-predictions) at www.microprediction.com (maybe jump to 
    https://www.microprediction.com/python-4). Basically what you need to do is set up a cron job or similar that periodically grabs a live
    data point and publishes it. First:

    
           pip install microprediction
           from microprediction import new_key
           write_key = new_key(difficulty=12)  # <--- Takes a long time, sorry
           
Then your regular job can do the following: 
         
           from microprediction import MicroWriter
           writer = MicroWriter(write_key=write_key)
           writer.set(name='my_own_stream.json',value=3.14157)  # <--- Must end in .json 

This will create a stream like [airport short term parking](https://www.microprediction.org/stream_dashboard.html?stream=airport-ewr-short-term_parking_a_b_c) and a bunch of
hungry time-series algorithms will come to it. The remainder of this note deals only with skater creation. 

## On creating a skater function (batch edition)

Want to shove a batch-oriented model in for the purpose (mostly) of comparison against faster things? No problem. The "modern" way is illustrated by 
[skautoarima](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skautoarima.py) and it uses the 
[batch_skater_factory](https://github.com/microprediction/timemachines/blob/a3430520b04807026055d9f2ba2379481ef3d641/timemachines/skatertools/batch/batchskater.py#L8). 

## On creating a skater function ...
Now if you want more freedom that that...

   - Read the [README](https://github.com/microprediction/timemachines) to understand what a skater is and what it is not. The good news is that a skater
   is merely a function. The bad news is that the function must accomodate a few conventions. 
   
   
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
   use packages that are designed for more one-off tabular use - since it is helpful to be able to benchmark fast skaters against slow ones. However I would suggest
   trying out some of the packages in the "online" section of the package list (see [Popular Python Time-Series Packages](https://www.microprediction.com/blog/popular-timeseries-packages)). For
   instance state space models or online libraries like river seem promising.  
   
   
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

  - Hyper-Parameter space:
     - A float *r* in (0,1). 
     - This package provides functions *to_space* and *from_space*, for expanding to R^n using space filling curves, so that the callee's (hyper) parameter optimization can still exploit geometry, if it wants to.   
     
See [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) or file an issue if anything offends you greatly. 
 
## On testing a skater function 

At minimum, please add a test script like [tests/simple/test_simple.py](https://github.com/microprediction/timemachines/blob/main/tests/simple/test_simple.py). This one uses a canned
time-series of hospital wait times. 

    from timemachines.skaters.simple.movingaverage import precision_ema_ensemble, aggressive_ema_ensemble
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error

    SIMPLE_TO_TEST = [ precision_ema_ensemble, aggressive_ema_ensemble ]

    def test_ensemble():
        for f in SIMPLE_TO_TEST:
           err = hospital_mean_square_error(f=f, k=5, n=150)

Please ensure this test runs quickly. 

## On hyper-optimizing a skater function 

This is up to you. If you need real-world data, you have it. Peruse the [knowledge center](https://www.microprediction.com/knowledge-center) or jump to this [notebook](https://github.com/microprediction/microprediction/blob/master/notebook_examples/Python_Module_3_Getting_History.ipynb) showing
how to retrieve recently created live data. As note, I don't care too much for stale historical data and you're advised to use the fresh bait to avoid over-fitting and sneaky kinds of self-deception that arise 
over time. 

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

This part should get your excited, and it is pretty automatic. Of course, you want to see your skater climb the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html). The Elo
ratings are a by-product of regular testing of the package. I use Elo ratings because they are incremental and we can avoid putting strain on the servers provided by our friends at GitHub who support
open source work like this. You don't need to do much for this, merely include a list of 'bound' skaters (i.e. no hyper-param *r* expected) in a fashion similar to what I've done for 
[allpmdskaters](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/dlm/alldlmskaters.py). Finally, you can reference this when making a change to 
[skaters/allskaters.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/allskaters.py) which is the list of all skater lists. See the previous comment about not assuming
existence of a package, however. Your list should be empty if your skater uses the package infrequently_maintained_and_certain_to_crash_on_m1 and the user has not pip installed that. 


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
