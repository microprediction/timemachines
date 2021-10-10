# Time-machines contributors guide

You may ask yourself, "Well, how did I get here?" And you may ask yourself, "How do I work this?". 

Okay enough 80's rock. Chances are you're here because you reached out to connect on Linked-In and you have some manner of time-series or quantitative interest so I sent you an invite. And now you find yourself in a conspiracy aimed at upending artisan data science by unleashing a billion little reward-seeking algorithms into a micro-economy, commoditizing repeated quantitative tasks of all kinds, and making data scientists look like fools.  

Oh never mind. Let's start with "Hello, we hope you are interested in helping us create free open short term prediction of anything for anyone." 

## Goals 
 
The goals of the timemachines package (and other related repos):

   - Make available a large slew of fully autonomous univariate forecasting functions with sequence-to-sequence signature ("skaters")
   - Evaluate them continuously over fresh, live, diverse real-world timeseries (search "time-series Elo ratings")
   - Create better ones by stacking, composition and so forth. 
   - Stick some of them in "crawlers" than operate in real-time and predict time-series at www.microprediction.org where they play this [game](https://www.microprediction.com/blog/intro)
   - Thereby provide free high quality short-term prediction to anyone, anywhere, for any reason
   - Like [solving anomaly detection](https://www.microprediction.com/blog/anomaly) in a way that isn't totally stupid.
   - Or predicting stuff in a way that [doesn't take an eternity](https://www.microprediction.com/blog/fast)

## Contribution patterns

Crawling:

   0. Cut and paste a bash command to drive a "crawler". See [CONTRIBUTE_COMPUTE_LOCAL_ONE_LINE](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_LOCAL_ONE_LINE.md)
   1. Run a Python script on your local machine that drives a "crawler". See [CONTRIBUTE_COMPUTE_LOCAL](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_LOCAL.md)
   2. Run a Python script on a PythonAnywhere account that drives a "crawler". See [CONTRIBUTE_COMPUTE_PA](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_PA.md)
   3. Create any kind of crawler you like in any language (though Python is easier). See the [knowledge center](https://www.microprediction.com/knowledge-center) tutorials. 
   
Adding time-series functionality (skaters) to timemachines:

   5. See [CONTRIBUTE_BATCH_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_BATCH_STYLE_MODELS.md) to add new functionality using non-incremental methods.
   6. See [CONTRIBUTE_ONLINE_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_ONLINE_STYLE_MODELS.md) to add new functionality using incremental methods.
   

## Friday chats

If you would like to help, the various CONTRIBUTE_ files suggest very specific steps you can take. However the following
initial step is strongly recommended: 

  - Grab the slack invite from the [knowledge center](https://www.microprediction.com/knowledge-center)
  - Turn up to one of the informal chats we have every Friday noon EST. 
  - If that timezone is bad, maybe Tue 8pm EST. 
  
But if you are shy that's fine too. 
